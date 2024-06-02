from arch import *
from config import *
from dataset import *
from torch.utils.data import Dataset, DataLoader
import random
from torchvision.utils import save_image
import os
from collections import deque
import torch.optim as optim
import copy
from tqdm import tqdm
import json

class Agent():
    def __init__(self, load_DQN=False):
        self.EPS = EPS
        self.epsilon = EPSILON
       
        classifier = get_architecture(CLASSIFIER_ARCH, "cifar10").eval()
        classifier.load_state_dict(torch.load(r"Trained_model/classifier_cifar10.pth"))
        self.classifier = classifier.cuda()
        
        self.img_size = IMG_SIZE
        self.mask_size = MASK_SIZE
        self.max_iter = MAX_ITER
        self.num_masks = NUM_MASKS
        self.noise_sd = NOISE_SD
        self.num_grids = IMG_SIZE // MASK_SIZE
        self.mask_vectors = torch.randn(self.num_masks, *(3, self.mask_size, self.mask_size)) * NOISE_SD
        self.input_size = INPUT_DQN_SIZE
        self.output_size = OUTPUT_DQN_SIZE
        
        self.policy_net =  DQN(self.input_size, self.output_size).cuda().train()
        
        self.memory = deque(maxlen=1000)
        self.criterion = nn.MSELoss().cuda()
        self.BATCH_SIZE = BATCH_SIZE
        self.TARGET_UPDATE =TARGET_UPDATE
        self.GAMMA = GAMMA
        if load_DQN:
            print("Loading Trained model")
            self.policy_net.load_state_dict(torch.load(DQN_TRAINED))

        self.target_net = DQN(self.input_size, self.output_size).cuda()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=1e-6)
        train_dataset = get_dataset(DATASET)
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        self.action_lists = []
        self.reward_lists = []
        self.loss_lists = []
        

    def cal_l2(self, img, new_img):
        return torch.norm(img - new_img).item()


    
    
    def optimize_policy(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        transitions = random.sample(self.memory, self.BATCH_SIZE)
        
        batch = Transition(*zip(*transitions))
        batch_states = torch.stack(batch.state).cuda()
        batch_rewards = torch.stack(batch.reward).cuda()
        batch_next_states = torch.stack(batch.next_state).cuda()
        batch_actions = torch.stack(batch.action).cuda()

        state_action_values = self.policy_net(batch_states)
        state_action_values = torch.gather(input=state_action_values, dim=1, index=batch_actions.unsqueeze(0))

        next_state_values = self.target_net(batch_next_states)
        next_state_values = torch.max(next_state_values, dim=1)[0]  # Max Q-value for each sample
        
        expected_state_action_values = (next_state_values * self.GAMMA) + batch_rewards

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(0))
        
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()  
        # torch.cuda.empty_cache()

        return loss.cpu().item()
    
    def R(self, sensity, l2_norm): # custom
        # print(PD_img_noise, PD_img, l2_norm)
        # reward = -(1e-4)/abs(PD_img - PD_img_noise + 1e-5) + 1 / (l2_norm + 1e-3)
        reward = -  l2_norm +  sensity / 1e8
        return reward
        
    def map_index(self, x: int, y: int):
    # x: hàng, y: cột 
    
        x_min = x * (self.mask_size)
        y_min = y * (self.mask_size)

        return x_min, y_min
    
    def sensitity(self, GT_prob: float, GT_noise_prob: float):
        return  1e7 * abs(GT_prob - GT_noise_prob)
    

    def make_action(self, image, action):
        image = image.clone().squeeze(0)
        with torch.no_grad():
            mask = self.mask_vectors[0]
            i ,j = self.extract_i_j(action)
            x_min, y_min = self.map_index(i, j)
            part = image[:, x_min: x_min + self.mask_size, y_min: y_min + self.mask_size].clone()
            part = part + mask
            image[:, x_min: x_min + self.mask_size, y_min: y_min + self.mask_size] = part        
        return image
        
    def select_action_model(self, state, mode="train"):
        if random.random() < self.epsilon and mode=="train":
            action = random.randint(0, self.output_size - 1)
            return action
        with torch.no_grad():
            output = self.policy_net(state.cuda())
            action = torch.argmax(output).item()
        return int(action)
    
    def extract_i_j(self, indx):
        i = indx // self.num_grids
        j = indx % self.num_grids
        return i, j
    


                
    def cal_sensities(self, image, # current image state 
                      P_pred, # P of original pred(label) 
                      pred, 
                      l2_lists=None):
        
        mask = self.mask_vectors[0] 
        image = image.squeeze(0)
        sensities = []
        with torch.no_grad():
            for i in range(self.num_grids):
                for j in range(self.num_grids):
                    x_min, y_min = self.map_index(i, j)
                    image_clone = image.clone()
                    part = image_clone[:, x_min: x_min + self.mask_size, y_min: y_min + self.mask_size]
                    part = part + mask * 100
                    image_clone[:, x_min: x_min + self.mask_size, y_min: y_min + self.mask_size] = part
                    # save_image(image_clone, "view_postion_add_mask.png")
                    P_noise = self.classifier(image_clone.unsqueeze(0).cuda())
                    P_noise_pred = P_noise[0][pred]
                    s = self.sensitity(P_pred, P_noise_pred)
                    sensities.append(s)
            sensities = torch.tensor(sensities)

            # l2_lists = torch.flatten(torch.tensor(l2_lists)).cuda()
            return sensities
        # return state, sensities
                
    def train(self):
        self.policy_net.train()
        for ep in range(self.EPS):
            for (image, label, idx) in tqdm(self.train_loader):
                
                
                # view the proccess
                folder_image = os.path.join(TRACK_FOLDER, str(idx.item()))
                if not os.path.exists(folder_image):
                    os.mkdir(folder_image)
                save_image(image, os.path.join(folder_image, "origininal.png"))
                save_image(image, "origininal.png")
                
                image_clone = image.clone()
                P_GT = self.classifier(image.cuda()).cpu() # P_GT
                pred = torch.argmax(P_GT).item() # label = pred
                P_pred = P_GT[0][pred] # P_pred   
                print("\nLabel pred: ", pred)
                print("Probability Pred: ", P_pred)  
                                               
                # init
                # l2_lists = deque([0, 0, 0, 0], maxlen=4)
                actions_list = deque([0, 0, 0, 0], maxlen=4)
                sensities = self.cal_sensities(image_clone, P_pred, pred)
                features = self.classifier.features(image_clone.cuda()).view(-1).cpu()                
                current_state = torch.cat((features ,sensities, torch.flatten(torch.tensor(actions_list))))
                save_image(image_clone, "view_cal.png")
                
                for step in tqdm(range(self.max_iter + 1)):
                    if step == self.max_iter:
                        save_image(image_clone, os.path.join(folder_image, "not_sucess.png"))
                        break
                    
                    # take action
                    action = self.select_action_model(current_state) # index of grid                     
                    # self.action_lists.append(action)
                    
                    image_clone = self.make_action(image_clone, action)
                    save_image(image_clone, os.path.join(folder_image, f"process.png"))
                    save_image(image_clone, f"process.png")
                    
                    # reward
                    l2_norm = self.cal_l2(image_clone, image)
                    actions_list.append(action)
                    P_noise_pred = self.classifier(image_clone.unsqueeze(0).cuda()).cpu()[0]
                    s = self.sensitity(P_pred, P_noise_pred[pred])
                    reward = self.R(s, l2_norm)
                    print(reward)
                    self.reward_lists.append(reward.cpu().item())
                    # print("\nReward: ", reward)
                    
                    # observation
                    sensities = self.cal_sensities(image_clone, P_pred, pred)
                    features = self.classifier.features(image_clone.unsqueeze(0).cuda()).view(-1).cpu()        
                    
                    # compose state
                    next_state = torch.cat((features ,sensities, torch.flatten(torch.tensor(actions_list))))

                    # check success
                    if torch.argmax(P_noise_pred).item() != pred:
                        
                        save_image(image_clone, os.path.join(folder_image, f"noise_{l2_norm}.png"))
                        print("Success")
                        break
                    
                    # next_state.requires_grad_()
                    # current_state.requires_grad_()
                    
                    self.memory.append([current_state.cpu(), torch.tensor(action).cpu(), next_state.cpu() ,reward.cpu()])
                    
                    # optimization
                    current_state = next_state
                    loss_ = self.optimize_policy()
                    print("\nLoss: ", loss_)
                    self.loss_lists.append(loss_)
                    # torch.cuda.empty_cache()
                    
                    if step % self.TARGET_UPDATE == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                        with open(ACTION_TRACK, "w+") as f1, open(LOSS_TRACK, "w+") as f2, open(REWARD_TRACK, "w+") as f3:
                            json.dump(self.action_lists, f1)
                            json.dump(self.loss_lists, f2)
                            json.dump(self.reward_lists, f3)
                            print()
                         
                torch.save(self.policy_net.state_dict(), f"model_{ep}.pth")

        if self.epsilon > 0.1:
            self.epsilon -= 0.9 / 5

    def test(self):
        self.policy_net.load_state_dict(torch.load(DQN_TRAINED))
        
        self.policy_net.eval()
        for ep in range(self.EPS):
            for (image, label, idx) in tqdm(self.train_loader):
                folder_image = os.path.join(TEST_FOLDER, str(idx.item()))
                if not os.path.exists(folder_image):
                    os.mkdir(folder_image)
                save_image(image, os.path.join(folder_image, "origininal.png"))
                save_image(image, "origininal.png")
                
                image_clone = image.clone()
                P_GT = self.classifier(image.cuda()).cpu() # P_GT
                pred = torch.argmax(P_GT).item() # label = pred
                P_pred = P_GT[0][pred] # P_pred   
                print("\nLabel pred: ", pred)
                print("Probability Pred: ", P_pred)  
                                               
                # init
                # l2_lists = deque([0, 0, 0, 0], maxlen=4)
                actions_list = deque([0, 0, 0, 0], maxlen=4)
                sensities = self.cal_sensities(image_clone, P_pred, pred)
                features = self.classifier.features(image_clone.cuda()).view(-1).cpu()                
                current_state = torch.cat((features ,sensities, torch.flatten(torch.tensor(actions_list))))
                
                for step in tqdm(range(self.max_iter + 1)):
                    if step == self.max_iter:
                        save_image(image_clone, os.path.join(folder_image, "not_sucess.png"))
                        break
                    
                    # take action
                    action = self.select_action_model(current_state, "test") # index of grid                     
                    # self.action_lists.append(action)
                    
                    image_clone = self.make_action(image_clone, action)
                    save_image(image_clone, os.path.join(folder_image, f"process.png"))
                    save_image(image_clone, f"process.png")
                    
                    # reward
                    l2_norm = self.cal_l2(image_clone, image)
                    actions_list.append(action)
                    P_noise_pred = self.classifier(image_clone.unsqueeze(0).cuda()).cpu()[0]
                    s = self.sensitity(P_pred, P_noise_pred[pred])
                    reward = self.R(s, l2_norm)
                    self.reward_lists.append(reward.cpu().item())
                    print("\nReward: ", reward)
                    
                    # observation
                    sensities = self.cal_sensities(image_clone, P_pred, pred)
                    features = self.classifier.features(image_clone.unsqueeze(0).cuda()).view(-1).cpu()        
                    
                    # compose state
                    next_state = torch.cat((features ,sensities, torch.flatten(torch.tensor(actions_list))))

                    # check success
                    if torch.argmax(P_noise_pred).item() != pred:
                        
                        save_image(image_clone, os.path.join(folder_image, f"noise_{l2_norm}.png"))
                        print("Success")
                        break
                    
                    # next_state.requires_grad_()
                    # current_state.requires_grad_()
                    
                    
                    # optimization
                    current_state = next_state
    def inference(self, image):
        self.policy_net.load_state_dict(torch.load(DQN_TRAINED))
        
        image = transforms.ToTensor()(image).unsqueeze(0)
        
        output_folder = os.path.basename("test").split(".")[0]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        self.policy_net.eval()
        save_image(image, os.path.join(output_folder, "origininal.png"))
        
        image_clone = image.clone()
        P_GT = self.classifier(image.cuda()).cpu() # P_GT
        pred = torch.argmax(P_GT).item() # label = pred
        P_pred = P_GT[0][pred] # P_pred  
        P_lists = []
        print("\nLabel pred: ", pred)
        print("Probability Pred: ", P_pred)  
        
                                        
        # init
        # l2_lists = deque([0, 0, 0, 0], maxlen=4)
        actions_list = deque([0, 0, 0, 0], maxlen=4)
        sensities = self.cal_sensities(image_clone, P_pred, pred)
        features = self.classifier.features(image_clone.cuda()).view(-1).cpu()                
        current_state = torch.cat((features ,sensities, torch.flatten(torch.tensor(actions_list))))
        
        for step in tqdm(range(self.max_iter + 1)):
            if step == self.max_iter:
                save_image(image_clone,os.path.join(output_folder, "not_sucess.png"))
                break
            
            # take action
            action = self.select_action_model(current_state, "test") # index of grid                     
            print(action)
            # self.action_lists.append(action)
            
            image_clone = self.make_action(image_clone, action)
            save_image(image_clone, os.path.join(output_folder, f"process_{step}.png")) # os.path.join(output_folder, f"process_{step}.png")
            
            # reward
            l2_norm = self.cal_l2(image_clone, image)
            actions_list.append(action)
            P_noise_pred = self.classifier(image_clone.unsqueeze(0).cuda()).cpu()[0]
            P_lists.append(P_noise_pred)
            print("P_noise: ", P_noise_pred[pred]) 

            if torch.argmax(P_noise_pred).item() != pred: 
                save_image(image_clone, os.path.join(output_folder, f"success{l2_norm}.png"))
                print("Success")
                break
                        
            # observation
            sensities = self.cal_sensities(image_clone, P_pred, pred)
            features = self.classifier.features(image_clone.unsqueeze(0).cuda()).view(-1).cpu()        
            
            # compose state
            next_state = torch.cat((features ,sensities, torch.flatten(torch.tensor(actions_list))))
            current_state = next_state
# a = Agent()
# a.train()
# path = r"D:\Reforinment-Learing-in-Advesararial-Attack-with-Image-Classification-Model\Adversarial_Attack_deeplearning\Splits\5\9.png"
# image = Image.open(path)
# a.inference(image)
                        