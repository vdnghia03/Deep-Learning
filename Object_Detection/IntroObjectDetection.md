# Introduction to Object Detection in Deep Learning

### I. Mục tiêu của Video và Chuỗi Video Sắp tới

Mục tiêu của video này là để hiểu **Object Detection** là gì và cách nó hoạt động, đồng thời có được cái nhìn tổng quan về tiến trình lịch sử với các kiến trúc mô hình khác nhau.

Chủ đề tiếp theo sẽ được đề cập bao gồm:
*   **Intersection Over Union (IoU):** Cách để đánh giá chất lượng của các bounding box đầu ra.
*   **Non-Max Suppression (NMS):** Cách giải quyết vấn đề có nhiều dự đoán bounding box cho cùng một đối tượng.
*   **Mean Average Precision (mAP)**.
*   Xây dựng thuật toán **YOLO** (You Only Look Once) từ đầu bằng PyTorch.

### II. Định nghĩa và Khái niệm Cơ bản

Để hiểu Object Detection, chúng ta bắt đầu từ các nhiệm vụ đơn giản hơn:

| Nhiệm vụ | Mục tiêu | Đặc điểm |
| :--- | :--- | :--- |
| **Image Classification (Phân loại ảnh)** | Nói *vật gì* có trong ảnh. | Nhiệm vụ đơn giản nhất. |
| **Object Localization (Định vị đối tượng)** | Tìm *cái gì* và *ở đâu* **một đối tượng duy nhất** tồn tại trong ảnh (Ví dụ: Một con mèo + một bounding box). | Chỉ tập trung vào một đối tượng trong mỗi ảnh. |
| **Object Detection (Phát hiện đối tượng)** | Tìm *cái gì* và *ở đâu* **nhiều đối tượng** tồn tại trong ảnh. | Là trường hợp tổng quát nhất. |

### III. Cách Thực hiện Object Localization

Để thực hiện định vị đối tượng (Localization), ta mở rộng từ Phân loại ảnh:

1.  **Đầu vào:** Một ảnh được gửi qua một Mạng Tích chập (CNN) như VGG hoặc ResNet.
2.  **Đầu ra Phân loại:** Các nút đầu ra dự đoán xác suất cho các lớp khác nhau (ví dụ: xác suất là mèo, xác suất là chó).
3.  **Đầu ra Định vị:** Thêm **bốn nút bổ sung** tương ứng với tọa độ của bounding box cho đối tượng cụ thể đó.
4.  **Định nghĩa Bounding Box:** Cần ít nhất bốn điểm để xác định một bounding box.
    *   Cách phổ biến nhất là sử dụng hai điểm góc: **x1 y1** (góc trên bên trái) và **x2 y2** (góc dưới bên phải).
    *   Có thể dùng hai điểm để xác định một góc và hai điểm khác để xác định chiều cao và chiều rộng.
5.  **Hàm Mất mát (Loss):**
    *   Sử dụng **Cross Entropy Loss** cho dự đoán phân loại (ví dụ: mèo hay chó).
    *   Sử dụng **L2 Loss** hoặc **Mean Squared Error** (MSE) cho các tọa độ, vì chúng là các giá trị số thực.

### IV. Các Phương pháp Phát hiện Đối tượng (Object Detection)

Vấn đề lớn của Object Detection là không thể có một số lượng nút đầu ra cố định vì số lượng đối tượng trong ảnh có thể là tùy ý. Có nhiều cách tiếp cận khác nhau để giải quyết vấn đề này:

#### 1. Phương pháp Cửa sổ Trượt (Sliding Windows)

Đây là một trong những cách tiếp cận sớm nhất và là phần mở rộng tự nhiên của Localization.

*   **Cách thức:** Xác định trước một kích thước bounding box và trượt nó qua ảnh với một bước nhảy (stride) cụ thể.
*   **Xử lý:** Cắt (crop) khu vực được cửa sổ trượt bao phủ, đổi kích thước (ví dụ: 224x224) để phù hợp với đầu vào của CNN, sau đó gửi qua CNN để xem có đối tượng nào không.
*   **Vấn đề:**
    *   **Tính toán cao:** Cần rất nhiều tính toán (compute). Lý tưởng nhất là di chuyển bounding box chỉ một pixel mỗi lần, đòi hỏi phải chạy một lượng lớn các phần ảnh đã đổi kích thước qua mạng CNN khổng lồ (ResNet hoặc VGG).
    *   **Nhiều kích cỡ:** Cần chạy lại toàn bộ quá trình với nhiều kích cỡ bounding box khác nhau (cho các đối tượng ở gần hoặc xa).
    *   **Nhiều Bounding Box:** Thu được nhiều dự đoán bounding box cho cùng một đối tượng.
*   **Cải tiến:** Bài báo **Overfeed** cho thấy có thể triển khai Sliding Window trong mạng tích chập (Conv Net) mà không cần cắt thủ công từng phần và gửi chúng riêng lẻ, mặc dù vấn đề tính toán vẫn tồn tại.

#### 2. Mạng Dựa trên Vùng (Regional Based Networks - R-CNN)

Cách tiếp cận này nhanh chóng thay thế Sliding Windows.

*   **Cách thức (R-CNN gốc):**
    *   Có một ảnh đầu vào.
    *   Sử dụng thuật toán để trích xuất các **đề xuất vùng** (region proposals).
    *   Ban đầu, họ sử dụng **Selective Search** (một thuật toán xác định, không phải mạng nơ-ron) để trích xuất khoảng 2.000 bounding box tiềm năng.
    *   Đổi kích thước tất cả 2.000 vùng tiềm năng này về kích thước cố định (ví dụ: 224x224) và chạy qua một Mạng Tích chập.
    *   Thực hiện dự đoán lớp và có thêm đầu ra để điều chỉnh tiềm năng vị trí của bounding box ban đầu.
*   **Ưu điểm:** Giảm đáng kể số lượng ảnh phải gửi qua CNN (cố định 2.000 vùng) so với Sliding Windows. Thuật toán Selective Search đã lo việc xác định kích thước bounding box cho các vùng cắt.
*   **Sự phát triển:** R-CNN được tiếp nối bởi **Fast R-CNN** và **Faster R-CNN**. Ở cuối các bài báo này, cơ chế đề xuất vùng cũng được thay đổi để trở thành một mạng nơ-ron.
*   **Vấn đề:**
    *   **Vẫn chậm:** Mặc dù đã có Faster R-CNN, thuật toán này vẫn còn xa mới đạt được phát hiện đối tượng thời gian thực (real-time).
    *   **Quá trình hai bước phức tạp:** Phải có bước đề xuất vùng trước, sau đó mới xác định xem đó có phải là bounding box hay không.

#### 3. Thuật toán YOLO (You Only Look Once)

YOLO là cách tiếp cận đơn bước (single-step), end-to-end, nhằm giải quyết sự phức tạp của R-CNN.

*   **Ý tưởng:**
    *   Tách ảnh gốc thành một lưới S x S (ví dụ: 7x7).
    *   **Mỗi ô lưới** chịu trách nhiệm dự đoán cả hai điều sau: (a) liệu có một bounding box trong ô đó không, và (b) xác suất lớp cho ô cụ thể đó.
*   **Trách nhiệm Bounding Box:** Một ô lưới chịu trách nhiệm dự đoán bounding box nếu ô đó là **điểm trung tâm** của đối tượng.
*   **Vấn đề (với YOLO v1):** Việc xác định xem ô nào là trung tâm của đối tượng có thể khó, dẫn đến việc mạng nơ-ron cho ra nhiều bounding box khác nhau như là đầu ra.
*   **Sự phát triển:** YOLO là một thuật toán rất phổ biến với bốn phiên bản (v1, v2, v3, v4.......)
