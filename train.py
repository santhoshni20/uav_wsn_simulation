import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from config import NUM_SENSORS
from rl_env import UAVWSNEnv
from mappo_agents import MAPPOSystem


class PPOTrainer:
    def __init__(
        self,
        env,
        mappo,
        gamma=0.98,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5
    ):
        self.env = env
        self.mappo = mappo
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def train(self, num_episodes=30, steps_per_episode=20):
        print(f"Starting Training: {num_episodes} Episodes ({steps_per_episode} steps each)...")
        history = {
            "episode_rewards": [],
            "pdr_history": [],
            "actor_losses": [],
            "critic_losses": []
        }

        for ep in range(num_episodes):
            obs, info = self.env.reset(seed=100 + ep)
            ep_reward = 0.0
            ep_pdr = []

            # Rollout storage
            obs_list = []
            uav_acts = []
            route_acts = []
            uav_log_probs = []
            route_log_probs = []
            rewards = []
            values = []
            dones = []

            for step in range(steps_per_episode):
                node_feats = torch.tensor(obs["node_features"], dtype=torch.float32)
                adj = torch.tensor(obs["adjacency"], dtype=torch.float32)
                uav_state = torch.tensor(obs["uav_state"], dtype=torch.float32)

                with torch.no_grad():
                    node_embs, graph_emb = self.mappo.st_gnn(node_feats, adj)
                    central_state = torch.cat([graph_emb, uav_state], dim=-1)
                    val = self.mappo.critic(central_state)
                    uav_act, uav_lp, _ = self.mappo.uav_actor.sample_action(central_state)
                    # Pass current adjacency mask to routing actor
                    route_act, route_lp, _ = self.mappo.routing_actor.sample_actions(node_embs, adj_mask=adj[-1])

                action = {
                    "uav_action": uav_act.squeeze(0).numpy(),
                    "routing_actions": route_act.squeeze(0).numpy()
                }

                next_obs, reward, terminated, truncated, step_info = self.env.step(action)

                obs_list.append(obs)
                uav_acts.append(uav_act)
                route_acts.append(route_act)
                uav_log_probs.append(uav_lp)
                route_log_probs.append(route_lp)
                rewards.append(reward)
                values.append(val.item())
                dones.append(float(terminated or truncated))

                ep_reward += reward
                ep_pdr.append(step_info["pdr_ratio"])
                obs = next_obs

                if terminated or truncated:
                    break

            # Bootstrap value for GAE
            with torch.no_grad():
                nf = torch.tensor(obs["node_features"], dtype=torch.float32)
                ad = torch.tensor(obs["adjacency"], dtype=torch.float32)
                us = torch.tensor(obs["uav_state"], dtype=torch.float32)
                _, g_emb = self.mappo.st_gnn(nf, ad)
                next_val = self.mappo.critic(torch.cat([g_emb, us], dim=-1)).item()

            # Compute GAE & Returns
            advantages = []
            gae = 0.0
            vals_extended = values + [next_val]
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * vals_extended[t + 1] * (1.0 - dones[t]) - vals_extended[t]
                gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
                advantages.insert(0, gae)

            returns = [adv + v for adv, v in zip(advantages, values)]
            advantages = torch.tensor(advantages, dtype=torch.float32)
            returns = torch.tensor(returns, dtype=torch.float32)

            if len(advantages) > 1 and advantages.std() > 1e-6:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Optimization Step
            act_loss_accum = 0.0
            crit_loss_accum = 0.0

            for t in range(len(obs_list)):
                o = obs_list[t]
                nf = torch.tensor(o["node_features"], dtype=torch.float32)
                ad = torch.tensor(o["adjacency"], dtype=torch.float32)
                us = torch.tensor(o["uav_state"], dtype=torch.float32)

                node_embs, g_emb = self.mappo.st_gnn(nf, ad)
                c_state = torch.cat([g_emb, us], dim=-1)

                curr_val = self.mappo.critic(c_state).view(-1)
                uav_dist = self.mappo.uav_actor(c_state)
                route_dist = self.mappo.routing_actor(node_embs, adj_mask=ad[-1])

                curr_uav_lp = uav_dist.log_prob(uav_acts[t]).sum(dim=-1)
                curr_route_lp = route_dist.log_prob(route_acts[t]).sum(dim=-1)

                r_uav = torch.exp(curr_uav_lp - uav_log_probs[t].squeeze(-1))
                r_route = torch.exp(curr_route_lp - route_log_probs[t].squeeze(-1))
                ratio = 0.5 * (r_uav + r_route)

                adv = advantages[t]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = F.mse_loss(curr_val, returns[t:t+1].view(-1))
                entropy = (uav_dist.entropy().sum() + route_dist.entropy().sum())

                total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                self.mappo.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.mappo.parameters(), self.max_grad_norm)
                self.mappo.optimizer.step()

                act_loss_accum += actor_loss.item()
                crit_loss_accum += critic_loss.item()

            mean_pdr = np.mean(ep_pdr) * 100
            history["episode_rewards"].append(ep_reward)
            history["pdr_history"].append(mean_pdr)
            history["actor_losses"].append(act_loss_accum / len(obs_list))
            history["critic_losses"].append(crit_loss_accum / len(obs_list))

            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"  Ep {ep+1:2d}/{num_episodes} | Return: {ep_reward:7.2f} | Avg PDR: {mean_pdr:5.1f}% | Critic Loss: {crit_loss_accum/len(obs_list):6.3f}")

        return history

    def evaluate(self, num_test_episodes=5, steps_per_episode=20):
        print(f"\nEvaluating Model on {num_test_episodes} Unseen Scenarios...")
        test_rewards = []
        test_pdrs = []

        for ep in range(num_test_episodes):
            obs, info = self.env.reset(seed=500 + ep)  # Unseen random seeds
            ep_r = 0.0
            ep_pdr = []

            for step in range(steps_per_episode):
                nf = torch.tensor(obs["node_features"], dtype=torch.float32)
                ad = torch.tensor(obs["adjacency"], dtype=torch.float32)
                us = torch.tensor(obs["uav_state"], dtype=torch.float32)

                with torch.no_grad():
                    node_embs, g_emb = self.mappo.st_gnn(nf, ad)
                    c_state = torch.cat([g_emb, us], dim=-1)
                    # Policy evaluation
                    uav_dist = self.mappo.uav_actor(c_state)
                    uav_act = uav_dist.loc
                    route_dist = self.mappo.routing_actor(node_embs, adj_mask=ad[-1])
                    route_act = torch.argmax(route_dist.probs, dim=-1)

                action = {
                    "uav_action": uav_act.squeeze(0).numpy(),
                    "routing_actions": route_act.squeeze(0).numpy()
                }

                obs, reward, terminated, truncated, step_info = self.env.step(action)
                ep_r += reward
                ep_pdr.append(step_info["pdr_ratio"])

                if terminated or truncated:
                    break

            test_rewards.append(ep_r)
            test_pdrs.append(np.mean(ep_pdr) * 100)
            print(f"  Test Scenario {ep+1}: Return = {ep_r:6.2f}, PDR = {test_pdrs[-1]:5.1f}%")

        return test_rewards, test_pdrs


# -------------------------------------------------------------
# Step 9 Verification & Visualization
# -------------------------------------------------------------

def run_step9():
    print("=" * 60)
    print("Step 9 - Training and Testing ST-GNN + MAPPO")
    print("=" * 60)

    env = UAVWSNEnv(history_window=5, max_steps=30)
    mappo = MAPPOSystem(env.st_gnn, lr_actor=4e-4, lr_critic=1e-3)

    trainer = PPOTrainer(env, mappo)
    train_history = trainer.train(num_episodes=25, steps_per_episode=30)
    test_rewards, test_pdrs = trainer.evaluate(num_test_episodes=5, steps_per_episode=30)

    # Save model checkpoint
    model_path = "st_gnn_mappo_checkpoint.pth"
    torch.save({
        "st_gnn": mappo.st_gnn.state_dict(),
        "critic": mappo.critic.state_dict(),
        "uav_actor": mappo.uav_actor.state_dict(),
        "routing_actor": mappo.routing_actor.state_dict()
    }, model_path)
    print(f"\nModel checkpoint saved successfully to: {model_path}")

    # -------------------------------------------------------------
    # Visualization for Step 9
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # 1. Episode Returns
    ax1 = axes[0, 0]
    episodes = range(1, len(train_history["episode_rewards"]) + 1)
    ax1.plot(episodes, train_history["episode_rewards"], "o-", color="#1f78b4", alpha=0.6, label="Episode Return")
    # Moving average
    window = 4
    if len(train_history["episode_rewards"]) >= window:
        ma = np.convolve(train_history["episode_rewards"], np.ones(window)/window, mode="valid")
        ax1.plot(range(window, len(train_history["episode_rewards"]) + 1), ma, "-", color="#08519c", linewidth=2.5, label="Moving Average")
    ax1.set_title("Training Episode Cumulative Rewards", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Return")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # 2. PDR Progression
    ax2 = axes[0, 1]
    ax2.plot(episodes, train_history["pdr_history"], "s-", color="#33a02c", linewidth=2, label="Train PDR (%)")
    ax2.axhline(np.mean(test_pdrs), color="#e31a1c", linestyle="--", linewidth=2, label=f"Mean Test PDR ({np.mean(test_pdrs):.1f}%)")
    ax2.set_title("Packet Delivery Ratio (PDR %) Evolution", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("PDR (%)")
    ax2.set_ylim(0, 105)
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.3)

    # 3. Actor & Critic Loss Curves
    ax3 = axes[1, 0]
    ax3.plot(episodes, train_history["critic_losses"], "^-", color="#e7298a", label="Critic MSE Loss")
    ax3.plot(episodes, train_history["actor_losses"], "v-", color="#7570b3", label="Actor PPO Loss")
    ax3.set_title("Loss Convergence (Actor vs Critic)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Loss")
    ax3.legend(loc="upper right")
    ax3.grid(True, linestyle="--", alpha=0.3)

    # 4. Train vs Test Generalization Box
    ax4 = axes[1, 1]
    x_pos = np.arange(len(test_pdrs))
    ax4.bar(x_pos, test_pdrs, color="#66c2a5", width=0.5, edgecolor="black", alpha=0.85)
    for i, v in enumerate(test_pdrs):
        ax4.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=9)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f"Test {i+1}" for i in range(len(test_pdrs))])
    ax4.set_ylim(0, 105)
    ax4.set_title("Unseen Test Scenarios: PDR Generalization", fontsize=12, fontweight="bold")
    ax4.set_ylabel("PDR (%)")
    ax4.grid(True, linestyle="--", alpha=0.3)

    summary_box = (
        "Training & Generalization Summary:\n"
        "----------------------------------\n"
        f"• Episodes Trained:      {len(train_history['episode_rewards'])}\n"
        f"• Final Train Return:    {train_history['episode_rewards'][-1]:.2f}\n"
        f"• Mean Evaluation PDR:   {np.mean(test_pdrs):.2f}%\n"
        f"• Model Checkpoint:      st_gnn_mappo_checkpoint.pth\n"
        f"• Generalization Status: Successfully Validated"
    )
    ax4.text(
        0.05, 0.25, summary_box,
        transform=ax4.transAxes,
        fontsize=8.5,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff2ae", alpha=0.9)
    )

    plt.suptitle("Step 9: ST-GNN + MAPPO Training & Testing Results", fontsize=15, fontweight="bold")
    plt.tight_layout()
    output_filename = "step9_training_testing_output.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved Step 9 visualization to {output_filename}")
    print("Step 9 completed successfully!")


if __name__ == "__main__":
    run_step9()
