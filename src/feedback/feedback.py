import cv2
import numpy as np
import src.distances.distance as d


# lehet, hogy fel kell cserelni a x, y-t
"""
centerp - kozeppontok
frame_width - az eredeti frame szelessege, ezt skalazzuk le fix 720-ra majd
"""
def scaling(centerp, frame_width):
    max_vals = np.amax(centerp, axis=0)
    main_factor = 720/(frame_width)
    mini_factor_x = 360/(max_vals[0]+30)
    mini_factor_y = 480/(max_vals[1]+30)
    return main_factor, mini_factor_x, mini_factor_y


#kirajzoljuk a szines bb-s framet, mellette pedig bird-eye view nezet, es szoveg
"""
frame_image - a frame kepe
bbs - bounding-boxok tombje
centerp - kozeppontok tombje
dist - 1 meterre juto pixelek szama
nr - hanyadik frame, frame sorszama
frame_width - az eredeti frame szelessege
path - utvonal ahova a kepek kerulnek
"""
def feedback_image(frame_image, bbs, centerp, dist, nr, frame_width, path):
    sf, sfx, sfy = scaling(centerp, frame_width)
    locations = d.distance_calc(centerp, bbs, dist)
    risky = locations[0]
    critic = locations[1]
    idx = locations[2]
    red = (0, 0, 255)
    orange = (0, 165, 255)
    green = (0, 255, 0)

    # birds-eye view kép, adott merettel
    bew_img = np.zeros((480, 360, 3), np.uint8)

    # narancssárga, ha 1,5 - 2,0 m távolságban vannak - risky
    # piros,ha 1,5 m-nél közelebb vannak egymástól - critic
    for r in risky:
        cv2.line(bew_img, (int(r[0] * sfx), int(r[1] * sfy)), (int(r[2] * sfx), int(r[3] * sfy)), orange, 2)

    for c in critic:
        cv2.line(bew_img, (int(c[0] * sfx), int(c[1] * sfy)), (int(c[2] * sfx), int(c[3] * sfy)), red, 2)

    for cp in centerp:
        cv2.circle(bew_img, (int(cp[0] * sfx), int(cp[1] * sfy)), 4, (255, 255, 255), -1)

    # bb-k kirajzolása a megfelelő színnel az emberek köré
    # piros ha risky vagy critical tavolsagban van valakivel
    # egyébként zöld

    #inicalizalom, mert most nincs kep, sajnos
    frame_image = np.zeros((640, 640, 3), np.uint8)
    frame_image.resize(720, int(frame_image.shape[1]*sf), 3)

    for b in range(len(bbs)):
        color = green
        if b not in idx:
            color = red
        cv2.rectangle(frame_image, (int(bbs[b].x * sf), int(bbs[b].y * sf)), ((int(bbs[b].x * sf) + int(bbs[b].w * sf)),
                      (int(bbs[b].y * sf) + int(bbs[b].h * sf))), color)

    text = np.zeros((240, 360, 3), np.uint8)
    txt1 = 'Keep the distance: ' + str(len(idx))
    txt2 = 'Too close: ' + str(len(bbs) - len(idx))
    cv2.putText(text, txt1, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.6, (250, 250, 250), 1)
    cv2.putText(text, txt2, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 0.6, (250, 250, 250), 1)

    if frame_image.shape[1] < 720:
        black_i = np.zeros((720, 180, 3), np.uint8)
        concats_1 = cv2.vconcat([black_i, frame_image])
        concats_2 = cv2.vconcat([concats_1, black_i])
    else:
        concats_2 = frame_image

    concats = cv2.vconcat([bew_img, text])

    cv2.imwrite(path +'/img_'+str(nr)+'.jpg', cv2.hconcat([concats_2, concats]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def video_maker(filename, frames, fps):
    #width, height = frames.shape[1::-1]
    width, height = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filename, fourcc, fps, (height, width))

    cv2.waitKey(1)
    for frame in frames:
        writer.write(frame)