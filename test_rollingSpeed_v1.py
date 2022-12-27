from turtle import right
import cv2
import mediapipe as mp
import numpy as np
import math

class CircleQueue:
    def __init__(self,size,front,rear):
        self.front = front
        self.rear = rear
        self.size = size
        self.queue = [ None for _ in range(size) ]

    def isQueueFull(self) :
        if (self.rear == self.size-1) :
            return True
        else :
            return False

    def isQueueEmpty(self) :
        if (self.front == self.rear) :
            return True
        else :
            return False

    def enQueue(self,data) :
        if (self.isQueueFull()) :
            print("큐가 꽉 찼습니다.")
            return
        self.rear += 1
        self.queue[self.rear] = data

    def deQueue(self) :
        if (self.isQueueEmpty()) :
            print("큐가 비었습니다.")
            return None
        self.front += 1
        data = self.queue[self.front]
        self.queue[self.front] = None

        # 모든 사람을 한칸씩 앞으로 당긴다.
        while True :
            self.queue[self.front] = self.queue[self.front+1]
            self.front += 1
            if self.front == self.size - 1:
                self.queue[self.front] = None
                self.front = -1
                self.rear = self.size - 2
                break
        return 

    def peek(self) :
        if (self.isQueueEmpty()) :
            print("큐가 비었습니다.")
            return None
        return self.queue[self.front+1]
    
    def result(self,index):
        try:
            return self.queue[index]
        except:
            return None

    def last(self):
        return self.rear
## 전역 변수 선언 부분 ##

max_num_hands = 2

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

#"실험 원본 영상(22.12.04) .mp4"
# "실험 원본 영상 (회전 속도 측량 20.12.27).mp4"
cap = cv2.VideoCapture("실험 원본 영상 (회전 속도 측량 20.12.27).mp4")

w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


fourcc = cv2.VideoWriter_fourcc(*'DIVX') # *'DIVX' == 'D','I','V','X'

#꼭 덮어쓰기 안할라면 제발 영상명 바꿉시다 ㅜㅜ

out = cv2.VideoWriter('실험 회전 속도 측정 (22.12.27).avi', fourcc, 30, (w*2, h*2))

RoL = 0  # 왼손 오른손 판별 변수 (0 이면 왼손, 1이면 오른손)

POINT_DISTANCE = 7 # 시각화(범위) 설정함수

count = 1

front = rear = -1
right_queue = CircleQueue(w,front,rear)
left_queue = CircleQueue(w,front,rear)

if __name__ == "__main__" :
    
    while cap.isOpened():
        #데이터 시각화를 위한 빈 화면 생성
        right_cycle_img = np.full((h,w,3),255,np.uint8)
        left_cycle_img = np.full((h,w,3),255,np.uint8)
        logic_img = np.full((h,w,3),255,np.uint8)

        ret, original_img = cap.read()
        if not ret:
            break
        
        original_img = cv2.flip(original_img, 1)
        final_img = np.vstack([np.hstack([original_img,logic_img]),np.hstack([left_cycle_img,right_cycle_img])])

        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        result = hands.process(original_img)

        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            Left_result = []
            Right_result = []

            for res in result.multi_hand_landmarks:
                if  RoL== 0:    
                    for j, lm in enumerate(res.landmark):
                        Left_result.append((int(lm.x*w), int(lm.y*h)))

                        cv2.circle(logic_img,(int(lm.x*w), int(lm.y*h)),3,(0,255,0),-1,cv2.LINE_AA)
                    RoL += 1
                else:
                    for j, lm in enumerate(res.landmark):
                        Right_result.append((int(lm.x*w), int(lm.y*h)))
                        cv2.circle(logic_img,(int(lm.x*w), int(lm.y*h)),3,(255,0,0),-1,cv2.LINE_AA)
                    
                
                mp_drawing.draw_landmarks(original_img, res, mp_hands.HAND_CONNECTIONS)
            
            if len(Right_result)==21 and len(Left_result)==21 :
                result= []
                dot = (int((Right_result[0][0]-Left_result[0][0])/2)+Left_result[0][0], h)
                cv2.circle(logic_img,dot,5,(0,0,0),-1,cv2.LINE_AA)
                sum_r = 0
                for i in range(len(Right_result)):
                    
                    cv2.line(logic_img,dot,Right_result[i],(255,0,0))
                    cv2.line(logic_img,dot,Left_result[i],(0,255,0))   
                    
                    res_r=math.sqrt((Right_result[i][0]-dot[0])**2+(Right_result[i][1]-dot[1])**2)
                    res_l=math.sqrt((Left_result[i][0]-dot[0])**2+(Left_result[i][1]-dot[1])**2)
                    sum_r += (res_r+res_l) 
                    result.append(abs(res_r-res_l))
                    if i == 4 :

                        if right_queue.isQueueFull() and left_queue.isQueueFull() :
                            right_queue.deQueue();left_queue.deQueue()    
                        right_queue.enQueue(int(res_r)); left_queue.enQueue(int(res_l))
                    
                    
            
                # 손 크기에 따른 역치 부여
                scale =  (sum_r/(w*h))
                scale = round(scale,4)
                res_error_rate =sum(result)
                
                # sclae 적용
                res_error_rate = (res_error_rate /sum_r)*100

                # # 동기화 정도.
                res_syn_rate = 100 - res_error_rate
                res_syn_rate = round(res_syn_rate,2)

                
                
                
                
                if (right_queue.last() +1 )* POINT_DISTANCE < w:
                    for gra_index  in range(0,right_queue.last()+1):
                        if gra_index == 0:
                            continue
                        
                        gra_index_past = gra_index - 1    
                        dr_point_past = gra_index_past* POINT_DISTANCE
                        r_point_past =(dr_point_past,right_queue.result(gra_index_past))
                        l_point_past =(dr_point_past,left_queue.result(gra_index_past))

                        dr_point = gra_index* POINT_DISTANCE
                        r_point = (dr_point,right_queue.result(gra_index))
                        l_point = (dr_point,left_queue.result(gra_index))
                        
                        cv2.line(right_cycle_img,r_point_past,r_point,(255,0,0),thickness=3)
                        cv2.line(left_cycle_img,l_point_past,l_point,(0,255,0),thickness=3)
                else:
                    can_dr = int((right_queue.last()+1 ) - (w // POINT_DISTANCE))
                    for x_point , gra_index in enumerate(range(can_dr,right_queue.last()+1)):
                        if x_point == 0:
                            continue

                        x_point_past = x_point - 1
                        gra_index_past = gra_index - 1    
                        dr_point_past = x_point_past*POINT_DISTANCE
                        r_point_past =(dr_point_past,right_queue.result(gra_index_past))
                        l_point_past =(dr_point_past,left_queue.result(gra_index_past))

                        dr_point = x_point*POINT_DISTANCE
                        r_point = (dr_point,right_queue.result(gra_index))
                        l_point = (dr_point,left_queue.result(gra_index))
                        
                        cv2.line(right_cycle_img,r_point_past,r_point,(255,0,0),thickness=3)
                        cv2.line(left_cycle_img,l_point_past,l_point,(0,255,0),thickness=3)   
                


                text = "Synchronization: " + str(res_syn_rate) +"  Frame_count: " + str(count) 
                
                cv2.putText(original_img,text,(int(w/10)*1,int(h/10)*1),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),1,cv2.LINE_AA)
                cv2.putText(logic_img,text,(int(w/10)*1,int(h/10)*1),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),1,cv2.LINE_AA)
                cv2.putText(right_cycle_img,text,(int(w/10)*1,int(h/10)*1),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),1,cv2.LINE_AA)
                cv2.putText(left_cycle_img,text,(int(w/10)*1,int(h/10)*1),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),1,cv2.LINE_AA)
                
                cv2.putText(right_cycle_img,"Right_cycle",(int(w/10)*7,int(h/10)*1),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),1,cv2.LINE_AA)
                cv2.putText(left_cycle_img,"Left_cycle",(int(w/10)*7,int(h/10)*1),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),1,cv2.LINE_AA)
                final_img = np.vstack([np.hstack([original_img,logic_img]),np.hstack([left_cycle_img,right_cycle_img])])
                count +=1
            RoL = 0
        cv2.imshow('Visible video', final_img)
        out.write(final_img)

        if cv2.waitKey(1) == ord('q'):
            break