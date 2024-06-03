Outline slide:

# 1. Motivation
- Power của DNN, tuy nhiên thiếu sự bền bỉ

- Có nhiều cuộc AT mạnh ... Tuy nhiên thì những cái này là whitebox attack.

- Các cuộc tấn công như Blackbox practical in readline hơn

    -> Thử nghiệm ứng dung của RL trong Black box AT

    -> Một agent có thể thực hiện việc tấn công mô hình classification model

# 2. Methodlogy

##  Agent can conduct Blackbox-AT: 
- Khái niệm: Chỉ cần dựa vào input(bức ảnh) và output của mô hình ( phân bố xác suất)

- Insight: 
    
    - vùng "nhạy cảm": Nếu thay đổi nhỏ vùng đó của hình ảnh thì ảnh hưởng đến đầu ra của mô hình phân loại

        -> tấn công vào vùng nhạy cảm đó khả năng cao làm mô hình missclassified
    - Mục tiêu: 
        1. Gây ra thay đổi lớn trong đầu ra
        2. Hình ảnh bị tấn công nhìn không nhận thấy được bằng mắt thường

- DQN Agent definition: Thực hiện hành động + nhiễu vào những vùng nhạy cảm ( fixed grid) để gây ra missclassified

    - State: 
        - current_image features
        - sensities score of grid
    
    - Reward:
        - l2_norm(current_image, original_image): mục tiêu 1

        - diff in Probability score of Grount Truth label: mục tiêu 2

        return - l2_norm + diff 


- Baseline attack:
    - Hình


- Algorithm RL agent for training:

    ```
        Initialization: Q Network param, N noise_vectors với shape = grid
        
        Input: Dataset, Max_iter = 300
        
        Output: Optimize for DQN agent


        for image in Dataset do:
            state <- Initialization original state

            mask <- random choose from N noise_vectors
            
            i <- 0;

            new_image <- image


            while not misclassify or i < Max_iter:

                action = select_action_DQN_agent(state, mask)	

                new_image <- take_action(new_image)

                Calculate the Reward R
            
                update DQN agent

                state <- observation
    ```

- Problem: 

    - Nếu chỉ dựa vào L2_norm và sensities. Thì những vùng đã bị tấn công sẽ càng trở nên nhạy cảm ( dựa trên thực nghiệm), thì agent có xu hướng sẽ + vào chỗ đó liên tục (do chỗ đó cho sensities cao -> reward cao). Việc + vào một chỗ nhiều lần sẽ làm cho cuộc tấn công dễ bị phát hiện tuy **vẫn đảm bảo L2_norm nhỏ**.
        - Đưa hình minh họa con agent ngu

        -> Phải thêm thông tin để tránh việc cộng vào một chỗ

    - Solution
        
        - Thêm một thông tin về actions history

        - Reward: Nếu cộng vào chỗ đó với tần suất càng nhiều thì điểm càng giảm
            
            -  l2_norm + diff - actions_frequency[action]
         
# 3. Experiment

- Dataset
- Train, Test
## Model classifcation

## Deep Q network
- Architech
- input / output

## Blackbox-AT Evaluation
- AVG l2_norm, AVG step, AVG susscess

## số liệu
- DQN training
- AT results

## Demo
	