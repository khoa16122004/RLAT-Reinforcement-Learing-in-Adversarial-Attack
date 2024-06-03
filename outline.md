Outline slide:

# 1. Motivation
- Power của DNN, tuy nhiên thiếu sự bền bỉ

- Có nhiều cuộc AT mạnh ... Tuy nhiên thì những cái này là whitebox attack.

- Các cuộc tấn công như Blackbox practical in realife hơn (nói về balckbox)

    -> Thử nghiệm ứng dung của RL trong Black box AT

# 2. Phát biểu bài toán

- Input/Output, chỉ sự dụng đầu ra của mô hình

    -> Sử dụng một agent có thể thực hiện Blackbox-AT tốt nhất


- Insight: 

    - Mục tiêu ( tốt nhất ): 
        1. Gây ra thay đổi lớn trong đầu ra
        2. Hình ảnh bị tấn công nhìn không nhận thấy được bằng mắt thường

    - vùng "nhạy cảm": Nếu thay đổi nhỏ vùng đó của hình ảnh thì ảnh hưởng đến đầu ra của mô hình phân loại

        -> tấn công vào vùng nhạy cảm đó khả năng cao làm mô hình missclassified

# 3. Modeling

- Agent definition: Chọn ra những vùng nhạy cảm ( fixed grid) để gây ra missclassified để add nhiễu vào

- State: 
    - current_image features
    - sensities score of grid: Cho hình ảnh
        
        - Sử dụng một random mask + tất cả các grids vào và tính sensities score 

- Reward:
    - l2_norm(current_image, original_image): mục tiêu 1

    - diff in Probability score of Grount Truth label: mục tiêu 2

        return - l2_norm + diff 

# 4. Methodology

- DQN (map với modeling)
    - Cách hoạt động
    - Input, Output của cái mạng
    - Khúc optimizize: loss, backward, ngẫu nhiên

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
        
        - State: Thêm một thông tin về actions history

        - Reward: Thêm Nếu cộng vào chỗ đó với tần suất càng nhiều thì điểm càng giảm
            
            return l2_norm + diff - actions_frequency[action]
         
# 3. Experiment
- Cấu hình máy, number step
- Dataset
- Train, Test

## Model classifcation

## Blackbox-AT Evaluation
- AVG l2_norm, AVG step, AVG susscess

## số liệu
- AT results

## Demo

## Problem
	