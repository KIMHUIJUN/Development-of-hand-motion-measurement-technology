import cv2
import mediapipe as mp
import numpy as np

max_num_hands = 2
gesture = {
    0:'zero', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'promise', 7:'yeah', 8:'spiderman', 9:'V', 10:'ok',
    11:'small heart', 12:'heart half', 13:'thumb up', 14:'nails', 15:'234',
}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt('gesture_train_new.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


fourcc = cv2.VideoWriter_fourcc(*'DIVX') # *'DIVX' == 'D','I','V','X'
out = cv2.VideoWriter('output_feedback(22.11.24).avi', fourcc, 30, (w, h))


while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        hand_result = []

        for res in result.multi_hand_landmarks:
            
            joint = np.zeros((21, 3)) # joint: 21, x,y,z: 3
            for j, lm in enumerate(res.landmark): # 각 조인트마다 랜드마크 저장
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,] 15개의 각도?

            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            # Draw gesture result
            if idx in gesture.keys():
                org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
                cv2.putText(img, text=gesture[idx].upper(), org=(org[0], org[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                
                finger1 = (int(res.landmark[4].x * img.shape[1]), int(res.landmark[4].y * img.shape[0]))
                finger2 = (int(res.landmark[8].x * img.shape[1]), int(res.landmark[8].y * img.shape[0]))
                finger3 = (int(res.landmark[12].x * img.shape[1]), int(res.landmark[12].y * img.shape[0]))
                finger4 = (int(res.landmark[16].x * img.shape[1]), int(res.landmark[16].y * img.shape[0]))
                finger5 = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
                
                hand_result.append({
                    'hand': gesture[idx],
                    'org': org, # 손목
                    'finger1': finger1, # 엄지손가락 끝점
                    'finger2': finger2, # 검지손가락 끝점
                    'finger3': finger3,
                    'finger4': finger4,
                    'finger5': finger5
                })
                
                
#             # 손목의 좌표 표시
#             wrist = res.landmark[0]
#             wrist_x = float("{0:0.3f}".format(wrist.x))
#             wrist_y = float("{0:0.3f}".format(wrist.y))
#             coordinate = "x:{0}, y:{1}".format(wrist_x, wrist_y)
#             cv2.putText(img, coordinate, org=(int(res.landmark[0].x * img.shape[1]), 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            
            
        
            if len(hand_result) >= 2: # 손이 2개일 때
                
                if hand_result[0]['hand'] == hand_result[1]['hand']: # 양손의 제스처가 같을 때
                    
                    midline_x =  int((hand_result[0]['org'][0] + hand_result[1]['org'][0])/2) # 두 손목의 중앙선
                    cv2.line(img, (midline_x, 100),(midline_x, 1000), (0,0,0), 2)
                    
                    # Symmetry Score
                    score_1 = 100 - (abs(hand_result[0]['finger1'][0] + hand_result[1]['finger1'][0] - 2*(midline_x))
                                     + abs(hand_result[0]['finger2'][0] + hand_result[1]['finger2'][0] - 2*(midline_x))
                                     + abs(hand_result[0]['finger3'][0] + hand_result[1]['finger3'][0] - 2*(midline_x))
                                     + abs(hand_result[0]['finger4'][0] + hand_result[1]['finger4'][0] - 2*(midline_x))
                                     + abs(hand_result[0]['finger5'][0] + hand_result[1]['finger5'][0] - 2*(midline_x)))/img.shape[1]*100/5
                    # Horizontal Score
                    score_2 = 100 - (abs(hand_result[0]['org'][1] - hand_result[1]['org'][1])/img.shape[0]*100)
                    
                    cv2.putText(img, text="Symmetry Score: {}" .format(round(score_1,2)), org=(int(w/10)*2,int(h/10)*2), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.8, color=(0,0,255), thickness=1,lineType=cv2.LINE_AA)
                    cv2.putText(img, text="Horizontal Score: {}" .format(round(score_2,2)), org=(int(w/10)*6,int(h/10)*2), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.8, color=(0,0,255), thickness=1,lineType=cv2.LINE_AA)
                    print(score_1,score_2)
        
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Game', img)
    # out.write(img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hand_result
