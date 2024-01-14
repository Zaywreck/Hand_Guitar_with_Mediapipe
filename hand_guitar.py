import threading
 

import cv2
import mediapipe as mp
#sesleri oynatmak için playsound kütüphanesini kullanıyoruz
import pygame

# mediapipe kütüphanesini kullanarak elin parmaklarını tanımlıyoruz
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Kamerayı açıyoruz

pygame.mixer.init()

cap = cv2.VideoCapture(0)


# Gitar sesleri için yol tanımlamaları
index_sound_path_tip = "sounds/1E.mp3"
middle_sound_path_tip = "sounds/2B.mp3"
ring_sound_path_tip = "sounds/3G.mp3"
pinky_sound_path_tip = "sounds/4D.mp3"
index_sound_path_pip = "sounds/5A.mp3"
middle_sound_path_pip = "sounds/6E.mp3"
ring_sound_path_pip = "sounds/C_Chord.mp3"
pinky_sound_path_pip = "sounds/E_Chord.mp3"
index_sound_path_mcp = "sounds/Dm_Chord.mp3"
middle_sound_path_mcp = "sounds/D_Chord.mp3"
ring_sound_path_mcp = "sounds/C_Chord.mp3"
pinky_sound_path_mcp = "sounds/Dm_Chord.mp3"

is_sound_playing = False

def play_sound(sound_path):
    global is_sound_playing
    pygame.mixer.Sound(sound_path).play()
    is_sound_playing = False

def islem():
    global is_sound_playing
    # Kameradan gelen görüntüyü işlemek için döngü oluşturuyoruz
    with mp_hands.Hands(
        #örnek kodlarda vardı bu kısımlar o yüzden olduğu gibi aldım
        static_image_mode=False,
        # iki el olduğu takdirde kafayı yiyor o yüzden 1 el için ayarlıyoruz
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Elin parmaklarını tanımlıyoruz (kök noktlarını çıkardım çünkü çakışmalara yol açıyor diğer notalara ulaşılmıyor)
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
                    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
                    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
                    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

                    # Burada gerekli koşulların kontrolüyle ilgili sesi oynatıyoruz
                    if thumb_tip.x < index_tip.x + 0.02 and thumb_tip.y < index_tip.y + 0.02:
                        if not is_sound_playing:
                            is_sound_playing = True
                            threading.Thread(target=play_sound, args=(index_sound_path_tip,)).start()

                    if thumb_tip.x < middle_tip.x + 0.02 and thumb_tip.y < middle_tip.y + 0.02:
                        if not is_sound_playing:
                            is_sound_playing = True
                            threading.Thread(target=play_sound, args=(middle_sound_path_tip,)).start()

                    if thumb_tip.x < ring_tip.x + 0.02 and thumb_tip.y < ring_tip.y + 0.02:
                        if not is_sound_playing:
                            is_sound_playing = True
                            threading.Thread(target=play_sound, args=(ring_sound_path_tip,)).start()

                    if thumb_tip.x < pinky_tip.x + 0.02 and thumb_tip.y < pinky_tip.y + 0.02:
                        if not is_sound_playing:
                            is_sound_playing = True
                            threading.Thread(target=play_sound, args=(pinky_sound_path_tip,)).start()

                    if thumb_tip.x < index_pip.x + 0.02 and thumb_tip.y < index_pip.y + 0.02:
                        if not is_sound_playing:
                            is_sound_playing = True
                            threading.Thread(target=play_sound, args=(index_sound_path_pip,)).start()

                    if thumb_tip.x < middle_pip.x + 0.02 and thumb_tip.y < middle_pip.y + 0.02:
                        if not is_sound_playing:
                            is_sound_playing = True
                            threading.Thread(target=play_sound, args=(middle_sound_path_pip,)).start()

                    if thumb_tip.x < ring_pip.x + 0.02 and thumb_tip.y < ring_pip.y + 0.02:
                        if not is_sound_playing:
                            is_sound_playing = True
                            threading.Thread(target=play_sound, args=(ring_sound_path_pip,)).start()

                    if thumb_tip.x < pinky_pip.x + 0.02 and thumb_tip.y < pinky_pip.y + 0.02:
                        if not is_sound_playing:
                            is_sound_playing = True
                            threading.Thread(target=play_sound, args=(pinky_sound_path_pip,)).start()

            cv2.imshow('Hand Landmarks', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":

    t1 = threading.Thread(target=islem, args=())
 
    t1.start()
 
    t1.join()
 
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

cap.release()
cv2.destroyAllWindows()
