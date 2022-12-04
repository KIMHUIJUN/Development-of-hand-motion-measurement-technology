from turtle import right
import cv2
import mediapipe as mp
import numpy as np
import math
max_num_hands = 2


# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


cap = cv2.VideoCapture(0)
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


fourcc = cv2.VideoWriter_fourcc(*'DIVX') # *'DIVX' == 'D','I','V','X'
out = cv2.VideoWriter('output_Visible_Sync_small_hand(22.12.03).avi', fourcc, 30, (w, h))
out2 = cv2.VideoWriter('output_Logical_Sync_small_hand(22.12.03).avi', fourcc, 30, (w, h))
a = 0
count = 1
while cap.isOpened():
    img2 = np.full((h,w,3),255,np.uint8)
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        Left_result = []
        Right_result = []

        for res in result.multi_hand_landmarks:
            if a == 0:    
                for j, lm in enumerate(res.landmark):
                    Right_result.append((int(lm.x*w), int(lm.y*h)))
                    
                    cv2.circle(img2,(int(lm.x*w), int(lm.y*h)),3,(255,255,0),-1,cv2.LINE_AA)
                a+=1
            else:
                for j, lm in enumerate(res.landmark):
                    Left_result.append((int(lm.x*w), int(lm.y*h)))
                    cv2.circle(img2,(int(lm.x*w), int(lm.y*h)),3,(0,255,255),-1,cv2.LINE_AA)
                
            
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
        
        if len(Right_result)==21 and len(Left_result)==21 :
            result= []
            dot = (int((Right_result[0][0]-Left_result[0][0])/2)+Left_result[0][0], h)
            cv2.circle(img2,dot,5,(0,0,0),-1,cv2.LINE_AA)
            
            Right_x = set() 
            Right_y = set()
            Left_x = set()
            Left_y = set()

            for i in range(len(Right_result)):
                cv2.line(img2,dot,Right_result[i],(0,0,255))
                cv2.line(img2,dot,Left_result[i],(255,0,0))    
                res_r=math.sqrt((Right_result[i][0]-dot[0])**2+(Right_result[i][1]-dot[1])**2)
                res_l=math.sqrt((Left_result[i][0]-dot[0])**2+(Left_result[i][1]-dot[1])**2)
                
                result.append(abs(res_r-res_l))
                Right_x.add(Right_result[i][0]);Right_y.add(Right_result[i][1]);Right_x.add(Left_result[i][0]);Right_y.add(Left_result[i][1])
            
            Right_x_Lenght = max(list(Right_x)) - min(list(Right_x))  
            Right_y_Lenght = max(list(Right_y)) - min(list(Right_y))  
            # Left_x_Lenght = max(list(Left_x)) - min(list(Left_x))
            # Left_y_Lenght = max(list(Left_y)) - min(list(Left_y))
            # print(Right_x_Lenght,Right_y_Lenght,Left_x_Lenght,Left_y_Lenght)
            # print(max(list(Right_x)),min(list(Right_x)),max(list(Right_y)),min(list(Right_y)))
            # print(max(list(Left_x)),min(list(Left_x)),max(list(Left_y)), min(list(Left_y)))
            detectbox_x_lenght = Right_x_Lenght #+ Left_x_Lenght
            detectbox_y_lenght = Right_y_Lenght #+ Left_y_Lenght
            detectbox_size = detectbox_x_lenght * detectbox_y_lenght
            scale = 1 - (detectbox_size /(w*h))
            scale = round(scale,4)
            # print(scale)
            res_error_rate =sum(result)
            # print(res_error_rate)
            
            res_error_rate = (res_error_rate * scale) / 10
            # print(res_error_rate)
            
            # 동기화 정도
            res_syn_rate = 100 - res_error_rate
            res_syn_rate = round(res_syn_rate,2)
            text = "Synchronization: " + str(res_syn_rate) + "  error_rate :"+str(round(res_error_rate,2))
            cv2.putText(img2,text,(int(w/10)*1,int(h/10)*2),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(img,text,(int(w/10)*1,int(h/10)*2),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(img2,str(count),(int(w/10)*1,int(h/10)*1),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(img,str(count),(int(w/10)*1,int(h/10)*1),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),1,cv2.LINE_AA)
            count +=1
        a=0
    cv2.imshow('Visible video', img)
    cv2.imshow("Logical video", img2)
    out.write(img)
    out2.write(img2)

    if cv2.waitKey(1) == ord('q'):
        break