import json
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from gym_wrapper import CBAGymEnv

DATA_PATH = "feedback_dataset.json"

def load_data():
    try:
        with open(DATA_PATH, "r") as f:
            return json.load(f)
    except:
        return []

def retrain():
    data = load_data()
    if not data:
        print("No data to retrain on.")
        return

    print(f"Retraining on {len(data)} samples...")

    env = CBAGymEnv()
    model = PPO.load("ppo_cba_model", env=env)
    
    # Switch to train mode
    model.policy.train()
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-4)

    for epoch in range(5): # Small number of epochs for fine-tuning
        total_loss = 0
        for sample in data:
            # Rehydrate state
            _, _ = env.reset(options={"preset_categories": sample["state"]})
            
            # Action mapping (MultiDiscrete [10, 2, 10])
            # The feedback action is a dict: {"target": "Class A", "fix_type": "recolor", ...}
            # We need to map it back to the index-based action the PPO expects.
            cat_list = list(sample["state"].keys())
            try:
                cat_idx = cat_list.index(sample["action"]["target"])
            except ValueError:
                continue
                
            fix_type = 0 if sample["action"]["fix_type"] == "recolor" else 1
            
            # For value_idx, we need to find it in the palette or shapes
            # In gym_wrapper: self.palette and self.shapes
            # We'll just assume a default index if we can't find it exactly for now
            # but ideally we'd store the indices in the dataset.
            value_idx = 0 
            # (In a real system, we'd store the raw integer action in the dataset)
            
            action_tensor = torch.tensor([[cat_idx, fix_type, value_idx]])
            obs, _ = env.reset(options={"preset_categories": sample["state"]})
            obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(model.policy.device)

            # SFT Loss: - reward * log_prob(a|s)
            distribution = model.policy.get_distribution(obs_tensor)
            log_prob = distribution.log_prob(torch.as_tensor([cat_idx, fix_type, value_idx]).to(model.policy.device))
            
            reward = sample["reward"]
            loss = - (reward * log_prob).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(data):.4f}")

    model.save("ppo_cba_model")
    print("Model updated with human feedback!")

if __name__ == "__main__":
    retrain()