
## R-CNN: Giải thích Rõ ràng

### I. Mục Tiêu của Phát Hiện Đối Tượng (Object Detection)

Mục tiêu cơ bản của Object Detection là, khi nhận một ảnh đầu vào, mô hình (có thể coi là một hộp đen) cần phải xác định:

1.  **Đối tượng là gì** (phân loại).
2.  **Tọa độ hộp giới hạn (bounding box coordinates)** của đối tượng đó.

**Bounding Box** (Hộp giới hạn) là hình vuông hoặc hình chữ nhật được vẽ trên ảnh để định vị đối tượng. Nó được biểu diễn bằng 4 giá trị:
*   Tọa độ góc trên bên trái: $X$ và $Y$.
*   Chiều rộng (width) và Chiều cao (height).
*   ![OurGoal](/Object_Detection/assets/rcnn1.png)

### II. Hạn Chế của Kiến Trúc Cơ Bản (Object Localization)

Trước khi đến R-CNN, ta xét một kiến trúc mạng CNN cơ bản (dùng trong Phân loại ảnh) được điều chỉnh cho nhiệm vụ định vị đối tượng:

#### A. Kiến trúc Cơ bản
1.  **Đầu vào:** Ảnh được điều chỉnh kích thước thành hình vuông.
2.  **Trích xuất đặc trưng:** Ảnh đi qua các lớp Convolution và Pooling.
3.  **Vector đặc trưng:** Flatten các tensor hoặc sử dụng Global Average Pooling.
4.  **Phân loại:** Lớp Fully Connected Classifier để đưa ra điểm xác suất cho mỗi đối tượng.
   

#### B. Điều chỉnh cho Định vị (Localization)
Để làm nhiệm vụ phát hiện đối tượng/định vị, phần sau lớp Fully Connected được tách thành hai nhánh độc lập:
1.  **Nhánh Phân loại (Classifier):** Tính điểm lớp (sử dụng Cross Entropy Loss).
2.  **Nhánh Hồi quy Bounding Box:** Sử dụng một lớp Fully Connected khác để xuất ra tọa độ hộp giới hạn (sử dụng L2 Loss).
3.  **Loss Cuối cùng:** Tính bằng tổng có trọng số (weighted sum) của hai loss trên.

#### C. Nhược điểm
Kiến trúc này chỉ có một nhược điểm lớn: **Nó chỉ có thể phát hiện một đối tượng duy nhất** trong ảnh, vì nó chỉ xuất ra một bộ tọa độ hộp giới hạn. Kiến trúc này được sử dụng cho **Định vị Đối tượng (Object Localization)**, nơi chỉ có một đối tượng được mong đợi.

![ObjectLocalization](/Object_Detection/assets/rcnn2.png)

### III. Hạn Chế của Phương pháp Cửa Sổ Trượt (Sliding Window)

Để phát hiện nhiều đối tượng, phương pháp ban đầu đã được sử dụng là Cửa sổ trượt:
1.  **Cách thức:** Đặt một cửa sổ trượt (sliding window) ở vị trí góc trên bên trái, trích xuất vùng này.
2.  **Phân loại:** Vùng được trích xuất được đưa qua một bộ phân loại mạng CNN.
3.  **Đầu ra:** Mạng xuất ra $C + 1$ lớp ($C$ là các lớp đối tượng mong đợi, **$+1$ là nền (background)**, vì vùng được trích có thể không chứa đối tượng nào).
4.  **Lặp lại:** Cửa sổ trượt sau đó trượt đến mọi vị trí có thể.
   ![SlidingWindow](/Object_Detection/assets/rcnn3.png)

**Vấn đề:** Phương pháp này không khả thi về mặt tính toán vì số lượng vị trí cần kiểm tra là **cực kỳ lớn**:
*   Số vị trí tiềm năng là $(W - w + 1) \times (H - h + 1)$.
*   Thậm chí tồi tệ hơn, cần phải xét các hộp giới hạn với **tỷ lệ và tỷ lệ khung hình (aspect ratios and scales)** khác nhau.
*   Ví dụ: Nếu ảnh đầu vào là 224x224, cần phải kiểm tra khoảng **635 triệu hộp**, điều này không thể thực hiện được trong thời gian thực.

### IV. Kiến Trúc R-CNN (Region-based CNN)

**Ý tưởng cốt lõi** của R-CNN là thay vì khám phá mọi hộp giới hạn, ta sử dụng một thuật toán bên ngoài để **đề xuất một số vùng quan tâm** (region proposals).

#### A. Selective Search (Tìm kiếm Có chọn lọc)
*   R-CNN sử dụng thuật toán **Selective Search** cho nhiệm vụ này.
*   Selective Search có thể đề xuất khoảng **2.000 vùng** chỉ trong vòng 1 đến 2 giây.

#### B. Quy trình R-CNN
Quy trình hoạt động của R-CNN diễn ra theo các bước sau:
1.  **Đầu vào:** Ảnh gốc.
2.  **Selective Search:** Chạy Selective Search để tạo ra các đề xuất vùng (region proposals).
   ![SelectiveSearch](/Object_Detection/assets/rcnn4.png)
3.  **Trích xuất & Biến đổi:** Trích xuất các vùng đề xuất từ ảnh gốc, sau đó biến đổi chúng thành hình vuông (để phù hợp với đầu vào của module CNN).
4.  **CNN Module:** Các vùng được đưa qua module CNN để trich xuat dac trung
5.  **Nhánh Phan Loai (Classification):** Dua ra label cua vung proposal do
6.  **Nhánh Hồi quy (Bounding Box Regression):** Thêm một nhánh phụ trách việc **tinh chỉnh (tweak)** hộp giới hạn, giúp nó chính xác hơn (vì hộp do Selective Search tạo ra có thể không chính xác, ví dụ: thiếu tay áo hoặc tai ngựa).
    ![RCNN](/Object_Detection/assets/rcnn5.webp)

### V. Chi tiết Hồi quy Bounding Box (Bounding Box Regression)

Mục tiêu là tìm ra tọa độ hộp giới hạn cuối cùng ($B_x, B_y, B_h, B_w$) từ hộp đề xuất ban đầu ($P_x, P_y, P_h, P_w$) và một vector biến đổi ($T_x, T_y, T_h, T_w$) do mạng thần kinh tạo ra.

Quá trình này sử dụng hai phép toán chính:

1.  **Phép Tịnh tiến (Translation):** Điều chỉnh vị trí trung tâm ($X, Y$).
    *   $B_x$ được tạo ra bằng cách di chuyển $P_x$ một khoảng bằng $P_w T_w$.
    *   Tương tự, $B_y$ được tạo ra bằng cách di chuyển $P_y$ một khoảng bằng $P_h T_h$.

2.  **Phép Biến đổi Tỷ lệ Không gian Log (Log Space Scale Transfer):** Điều chỉnh kích thước ($W, H$).
    *   Chiều rộng $B_w$ được tính bằng cách nhân $P_w$ với **lũy thừa của $T_w$** ($e^{T_w}$).
    *   Sử dụng hàm lũy thừa ($e^x$) để đảm bảo chiều rộng luôn là giá trị dương, ngay cả khi $T_w$ là số âm.
    *   Chiều cao $B_h$ được tính tương tự.
    
    ![BoxRegression](/Object_Detection/assets/rcnn6.png)

### VI. Non-Maximal Suppression (NMS)

**Vấn đề:** Mô hình có thể xuất ra nhiều hộp giới hạn khác nhau, nhưng tất cả cùng chỉ vào **một đối tượng**.
**Mục tiêu:** Chọn ra hộp dự đoán tốt nhất và loại bỏ các hộp trùng lặp.

#### A. Intersection over Union (IoU)
IoU là tiêu chí được sử dụng để đo mức độ chồng lấn giữa hộp dự đoán và hộp Ground Truth (hộp được gán nhãn thủ công hoàn hảo).
*   **Công thức:** IoU = (Diện tích Giao nhau/Intersection) / (Diện tích Hợp lại/Union).
*   Hộp có IoU lớn hơn (gần với Ground Truth hơn) được coi là ứng cử viên tốt hơn.

#### B. Áp dụng NMS
NMS được sử dụng để:
*   Chọn ra hộp dự đoán có IoU cao nhất.
*   **Lưu ý:** NMS chỉ được áp dụng khi các hộp giới hạn đang tham chiếu đến **cùng một đối tượng**.

    ![NonMaximal](/Object_Detection/assets/rcnn7.png)

### VII. Đánh giá Mô hình: Mean Average Precision (mAP)

Mean Average Precision (mAP) là metric được đề xuất để đánh giá mô hình phát hiện đối tượng.

#### A. Ground Truth, Precision, và Recall
*   **Ground Truth (GT):** Hộp giới hạn hoàn hảo, được dán nhãn thủ công trước khi huấn luyện.
*   **Xác định dự đoán Đúng/Sai:** Một hộp dự đoán là **đúng** nếu IoU của nó với GT lớn hơn **0.5**; nếu không, nó là **sai**.
*   **Precision (Độ chính xác):** Tỷ lệ các dự đoán đúng trong tổng số các dự đoán đã thực hiện.
*   **Recall (Độ nhạy):** Tỷ lệ các hộp Ground Truth đã được dự đoán đúng trong tổng số các hộp Ground Truth.

#### B. Quy trình tính Mean Average Precision (mAP)
Quy trình tính mAP tập trung vào từng lớp đối tượng:

1.  **Sắp xếp:** Sắp xếp tất cả các hộp dự đoán (chỉ cho một lớp, ví dụ: "Dog") dựa trên điểm xác suất của chúng theo thứ tự giảm dần.
2.  **Tính tích lũy:** Xem xét từng hộp dự đoán theo thứ tự đã sắp xếp:
    *   Xác định hộp đó là Đúng hay Sai (dựa trên ngưỡng IoU > 0.5).
    *   Tính Precision và Recall mới sau khi xem xét hộp đó.
    *   Vẽ điểm Precision-Recall trên biểu đồ.
3.  **Average Precision (AP):** Sau khi xem xét tất cả các hộp, AP được tính bằng **diện tích dưới đường cong Precision-Recall**.
   ![NonMaximal](/Object_Detection/assets/rcnn8.png)
4.  **Mean Average Precision (mAP):** Lấy giá trị trung bình (average) của AP qua tất cả các lớp (ví dụ: (AP Dog + AP Cat) / 2).

**Lưu ý mở rộng:** Trong thực tế, mAP thường được tính bằng cách lấy trung bình các giá trị AP sử dụng các ngưỡng IoU khác nhau (ví dụ: IoU từ 0.5 đến 0.95, với bước 0.05).
