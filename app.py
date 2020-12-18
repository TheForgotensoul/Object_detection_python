import cv2
from pathlib import Path


def main(mode):
    threshold = 0.50
    img = ""
    class_file = "trained_models/coco.names"
    with open(class_file, "rt") as f:
        Class_names = f.read().split("\n")

    config_file = "trained_models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    weight_file = "trained_models/frozen_inference_graph.pb"

    net = cv2.dnn_DetectionModel(weight_file, config_file)
    net.setInputSize((320, 320))
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    def run():
        print('Press "q" to exit')
        class_ids, confidence, bbox = net.detect(img, threshold)
        # print(class_ids, bbox)
        if len(class_ids) != 0:
            for classId, confs, box in zip(class_ids.flatten(), confidence.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, Class_names[classId - 1], (box[0] + 5, box[1] + 15),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(img, str(round(confs * 100, 2)), (box[0] + len(Class_names[classId - 1] * 13), box[1] + 15),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("Output", img)

    if mode == 1:
        sc = int(input("Enter the webcam number (default = 0): "))
        if sc == "":
            sc = 0
        cap = cv2.VideoCapture(sc)
        cap.set(3, 640)
        cap.set(4, 480)
        cap.set(10, 100)
        while True:
            success, img = cap.read()
            run()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
    elif mode == 3:
        sc = Path(input("Provide the location of the file: "))
        cap = cv2.VideoCapture(f"{sc}")
        cap.set(3, 640)
        cap.set(4, 480)
        cap.set(10, 100)
        while True:
            success, img = cap.read()
            run()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
    else:
        threshold = 0.60
        sc = Path(input("Provide the location of the file: "))
        img = cv2.imread(f"{sc}")
        run()
        if cv2.waitKey(0) & 0xFF == ord("q"):
            cv2.destroyAllWindows()


try:
    print("""
    Available mode of inputs: 1 > live streaming from a web cam
                              2 > Images
                              3 > videos
    """)
    mde = int(input("Mode of Input: "))
    main(mde)

except Exception as e:
    print(e)
