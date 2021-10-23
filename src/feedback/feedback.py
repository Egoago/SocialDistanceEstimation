import cv2
import numpy as np
import src.distances.distance as d


# kirajzolja egy fekete képre a birds-eye view pontokat
# kirajzolja a képre megfelelő színnel az emberek bb-it
def feedback_image(img, bbs, centerp):
    locations = d.distance_calc(centerp)
    risky = locations[0]
    critic = locations[1]
    idx = locations[2]
    red = (0, 0, 255)
    orange = (0, 165, 255)
    green = (0, 255, 0)

    # birds-eye view kép
    bew_img = np.zeros((640, 640, 3), np.uint8)

    # narancssárga, ha 1,5 - 2,0 m távolságban vannak - risky
    # piros,ha 1,5 m-nél közelebb vannak egymástól - critic
    for r in risky:
        cv2.line(bew_img, (r[0], r[1]), (r[2], r[3]), orange, 2)

    for c in critic:
        cv2.line(bew_img, (c[0], c[1]), (c[2], c[3]), red, 2)

    for cp in centerp:
        cv2.circle(bew_img, cp, 4, (255, 255, 255), -1)

    cv2.imwrite('bew_images/bew_img_1.jpg', bew_img)

    # bb-k kirajzolása a megfelelő színnel az emberek köré
    # piros ha risky vagy critical tavolsagban van valakivel
    # egyébként zöld
    # alul kiírjuk, hogy hány ember tartja be a távolságot és hányan nem
    img = np.zeros((640, 640, 3), np.uint8)
    for b in range(len(bbs)):
        if b not in idx:
            cv2.rectangle(img, (bbs[b].x, bbs[b].y), (bbs[b].x + bbs[b].w, bbs[b].y + bbs[b].h), red)
        else:
            cv2.rectangle(img, (bbs[b].x, bbs[b].y), (bbs[b].x + bbs[b].w, bbs[b].y + bbs[b].h), green)

    img2 = np.zeros((200, 640, 3), np.uint8)
    txt1 = 'Be tartja a tavolsagot: ' + str(len(idx))
    txt2 = 'Veszelyben van: ' + str(len(bbs) - len(idx))
    cv2.putText(img2, txt1, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.6, (250, 250, 250), 1)
    cv2.putText(img2, txt2, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 0.6, (250, 250, 250), 1)

    cv2.imwrite('bb_images/img_1.jpg', cv2.vconcat([img, img2]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
