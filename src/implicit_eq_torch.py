import torch

torch.manual_seed(0)


# ====== 问题设置：y = f(y, x; θ) ======
class F(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn(d, d) * 0.3)  # 收敛更稳
        self.B = torch.nn.Parameter(torch.randn(d, d) * 0.3)
        self.b = torch.nn.Parameter(torch.zeros(d))

    def forward(self, y, x):
        # f(y,x) = tanh(Ay + Bx + b)
        return torch.tanh(y @ self.A.T + x @ self.B.T + self.b)


# ====== 0) 固定点求解器（迭代期间完全 no-grad） ======
def solve_fixed_point_no_grad(f, x, y0=None, max_iter=100, tol=1e-6):
    y = torch.zeros_like(x) if y0 is None else y0.detach().clone()
    with torch.no_grad():
        for _ in range(max_iter):
            y_next = f(y, x)  # 不记录图
            if (y_next - y).abs().max().item() < tol:
                y = y_next
                break
            y = y_next
    return y.detach()


# ====== 1) 只解不反传（演示报错的根因与正确写法） ======
def demo_no_grad_only():
    print("\n[Demo-1] 只解不反传")
    d = 4
    f = F(d)
    x = torch.randn(3, d)
    y_star = solve_fixed_point_no_grad(f, x)
    loss = ((y_star - 0.0) ** 2).mean()
    print("loss.item()=", float(loss))  # 不要 backward()，因为整个图都被切断了


# ====== 2) 迭代无梯度，但仍训练参数（单步接图） ======
def demo_single_step_attach():
    print("\n[Demo-2] 迭代无梯度，但参数可学（单步接图）")
    d = 4
    f = F(d)
    opt = torch.optim.Adam(f.parameters(), lr=1e-2)
    for step in range(50):
        x = torch.randn(32, d)
        target = torch.zeros_like(x)

        # (a) 先 no-grad 求 y*
        y_star = solve_fixed_point_no_grad(f, x)

        # (b) 再用“一步 f”把图接上（不会回放迭代）
        y_for_loss = f(y_star, x)  # 仅此一步有 grad_fn
        loss = torch.nn.functional.mse_loss(y_for_loss, target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 10 == 0:
            print(f"step {step:02d} | loss {loss.item():.6f}")


# ====== 3) 真·隐式微分：自定义 autograd.Function ======
# 反向传播需要解：(I - J_y)^T * u = grad_y L
# 我们用简单的固定点/Neumann 近似来求 u（也可换成共轭梯度）。
class ImplicitSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module_f, y0, max_iter, tol):
        # 前向：no-grad 迭代拿到 y*
        with torch.no_grad():
            y_star = solve_fixed_point_no_grad(
                module_f, x, y0=y0, max_iter=max_iter, tol=tol
            )
        # 保存需要的对象/张量用于 backward
        ctx.module_f = module_f
        ctx.save_for_backward(x, y_star)
        ctx.max_iter = max_iter
        ctx.tol = tol
        return y_star

    @staticmethod
    def backward(ctx, grad_y):  # grad_y = ∂L/∂y*
        module_f = ctx.module_f
        x, y_star = ctx.saved_tensors
        max_iter = ctx.max_iter
        tol = ctx.tol

        # 我们需要求解 u 满足：(I - J_y(y*,x;θ))^T u = grad_y
        # 用固定点迭代：u_{k+1} = grad_y + J_y(y*,x;θ)^T u_k
        # 其中 J_y^T u 通过一次反向 AD 实现（向量-雅可比积）。
        u = torch.zeros_like(grad_y)
        with torch.enable_grad():
            y_star_req = y_star.detach().requires_grad_(True)
            for _ in range(max_iter):
                # 计算 J_y^T u_k：对 y_star 的扰动方向 u，算 vjp
                f_y = module_f(y_star_req, x)  # 构图
                vjp = torch.autograd.grad(
                    outputs=f_y,
                    inputs=y_star_req,
                    grad_outputs=u,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )[0]
                u_next = grad_y + vjp
                if (u_next - u).abs().max().item() < tol:
                    u = u_next
                    break
                u = u_next

            # 现在有了 u，继续把梯度传给参数 θ 和输入 x
            # 需要 J_θ^T u 和 J_x^T u（向量-雅可比积）
            f_y = module_f(y_star_req, x)
            # 对参数的 vjp
            params = [p for p in module_f.parameters() if p.requires_grad]
            grads_params = torch.autograd.grad(
                outputs=f_y,
                inputs=params,
                grad_outputs=u,
                retain_graph=False,
                allow_unused=True,
            )
            # 对 x 的 vjp
            grad_x = torch.autograd.grad(
                outputs=f_y,
                inputs=x,
                grad_outputs=u,
                retain_graph=False,
                allow_unused=True,
            )[0]

        # 将得到的梯度填回
        # backward 的返回与 forward 的输入一一对应：(x, module_f, y0, max_iter, tol)
        # 只有 x 与 module_f(参数)可导；y0/max_iter/tol 都不可导，需返回 None。
        # 但 module_f 不是 Tensor，参数梯度已经通过上面的 vjp 累加在 .grad 中，
        # 这里对第二个返回位填 None 即可。
        # 注意：上面的对 params 的 grad 只是计算出来，还没加到 .grad，需要手动加：
        for p, g in zip(params, grads_params):
            if g is not None:
                if p.grad is None:
                    p.grad = g.detach()
                else:
                    p.grad = p.grad + g.detach()

        return grad_x, None, None, None, None


# 供外部调用的“隐式层”
class ImplicitLayer(torch.nn.Module):
    def __init__(self, f: torch.nn.Module, max_iter=50, tol=1e-6):
        super().__init__()
        self.f = f
        self.max_iter = max_iter
        self.tol = tol

    def forward(self, x, y0=None):
        if y0 is None:
            y0 = torch.zeros_like(x)
        return ImplicitSolve.apply(x, self.f, y0, self.max_iter, self.tol)


def demo_true_implicit_diff():
    print("\n[Demo-3] 真·隐式微分（IFT 近似 via VJP fixed-point）")
    d = 4
    f = F(d)
    layer = ImplicitLayer(f, max_iter=50, tol=1e-6)
    opt = torch.optim.SGD(f.parameters(), lr=5e-3)

    for step in range(50):
        x = torch.randn(32, d)
        target = torch.zeros_like(x)

        y_star = layer(x)  # 前向：no-grad 求解；反向：隐式 VJP 求解
        loss = torch.nn.functional.mse_loss(y_star, target)

        opt.zero_grad()
        # 注意：ImplicitSolve.backward 会把参数梯度直接加到 p.grad 上
        loss.backward()
        opt.step()

        if step % 10 == 0:
            print(f"step {step:02d} | loss {loss.item():.6f}")


# ====== 运行全部演示 ======
if __name__ == "__main__":
    demo_no_grad_only()
    demo_single_step_attach()
    # demo_true_implicit_diff()
