a_seq = torch.randn(H, action_dim, requires_grad=True)

for i in range(optimizer_steps):
    s_pred = rollout(f_theta, s_t, a_seq)
    cost = sum(cost_fn(s_pred[t], goal_pose) for t in range(H))
    
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()
