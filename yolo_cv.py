import rclpy 
from rclpy.node import Node
import cv2 
from ultralytics import YOLO

class YoloDetectionNode(Node):
    def __init__(self):

        super().__init__('yolo_node')

        #Yolo'nun Nano modelini kullanıyoruz 
        self.model = YOLO("yolov8n.pt")
        #Bilgisayarın varsayılan kamerasını bir veri akışı olarak başlatır.
        self.cap = cv2.VideoCapture(0)

        #0.03 saniyede bir frame_process fonksiyonunu çalıştır
        self.timer = self.create_timer(0.03 , self.frame_process)

    
    def frame_process(self):
        ret,frame = self.cap.read()
        if ret:
            results = self.model(frame,device=0,stream=True)
            #Goruntuyu al , gpuda yap , akış keasintisiz olsun

            # Generator olduğu için for döngüsüyle içindekini alıyoruz
            for r in results:
                # Kutuları ve etiketleri çizdiriyoruz
                annotated_frame = r.plot()

            #Üzerine kutular çizilen görüntüyü kullanıcı bilgisayarında bir pencere olarak açar
            cv2.imshow("YOLOv8 ROS2 Node", annotated_frame)
            #1ms lik ekranın yenilenmesi. Kod çökmesin diye
            cv2.waitKey(1)


    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()