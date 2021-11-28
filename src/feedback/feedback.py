import cv2
import numpy as np
from typing import List

from src.tracking import Person
from src.projection import project, opengl2opencv, back_project, opencv2opengl

from src.distances import calc_dist as calc


def feedback_image(camera, img_size, image: np.ndarray, people: List[Person], settings):
    # birds-eye view image
    bew_img = np.zeros((int(img_size[1] * 1.5), int(img_size[0] * 1.5), 3), np.uint8)

    centerp = []  # 2d coords center points
    cps = []  # pixel coords center points

    black = (0, 0, 0)
    white = (255, 255, 255)
    red = (255, 0, 0)
    orange = (255, 165, 0)
    green = (0, 255, 0)

    # the default bb color is white
    for person in people:
        person.color = white
        if camera is not None:
            c = back_project(np.array(opencv2opengl(person.bbox.bottom(), img_size[1])),
                             camera)
            centerp.append([(c[0]), (c[2])])
            center = person.bbox.x + person.bbox.w // 2, person.bbox.y + person.bbox.h // 2
            cps.append(center)
            person.color = black

    if camera is not None and len(centerp) != 0:
        scale = 10
        coordx = 0.5 * img_size[0] * scale
        coordy = -0.5 * img_size[1] * scale

        locations = calc.calc_dist(centerp, cps)
        risky_idx = locations[0]
        critic_idx = locations[1]
        idx = locations[2]
        risky = locations[3]
        critic = locations[4]

        for i in idx:
            people[i].color = green
        for ri in risky_idx:
            people[ri].color = orange
        for ci in critic_idx:
            people[ci].color = red
        for r in risky:
            cv2.line(image, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), orange, 3)
            cv2.line(bew_img, (int((r[5] + coordx) / scale), int((r[4] + coordy) / scale)),
                     (int((r[7] + coordx) / scale), int((r[6] + coordy) / scale)), orange, 15)
        for c in critic:
            cv2.line(image, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), red, 3)
            cv2.line(bew_img, (int((c[5] + coordx) / scale), int((c[4] + coordy) / scale)),
                     (int((c[7] + coordx) / scale), int((c[6] + coordy) / scale)), red, 15)
        for cp in centerp:
            cv2.circle(bew_img, (int((cp[1] + coordx) / scale), int((cp[0] + coordy) / scale)), 20, (255, 255, 255),
                       -1)

    for person in people:
        if settings.get('display_bounding_boxes'):
            top_left, bottom_right = person.bbox.corners()
            cv2.rectangle(image, top_left, bottom_right, person.color, 2)

        if camera is not None and settings.get('display_proximity'):
            res = 20
            radius = 1500
            center = back_project(np.array(opencv2opengl(person.bbox.bottom(), img_size[1])),
                                  camera)
            pixels = []
            for i in np.linspace(0, 2 * np.pi.real, res):
                point = center + np.array([np.cos(i), 0, np.sin(i)], dtype=float) * radius
                pixel = opengl2opencv(tuple(project(point, camera)[0]), img_size[1])
                pixels.append(pixel)
            cv2.polylines(image, np.int32([pixels]), True, white, 1)

        if settings.get('display_centers'):
            center = person.bbox.x + person.bbox.w // 2, person.bbox.y + person.bbox.h // 2
            cv2.circle(image, center, 6, person.color, 8)

    image_resized = cv2.resize(image, (960, 540))
    bew_img_resized = cv2.flip(cv2.resize(bew_img, (320, 180)), 0)

    # drawing the bew image on the frame
    row, col, ch = bew_img_resized.shape
    overlay = cv2.addWeighted(image_resized[328:328 + row, 32:32 + col], 0.5, bew_img_resized, 0.5, 0)
    image_resized[328:328 + row, 32:32 + col] = overlay

    return image_resized
