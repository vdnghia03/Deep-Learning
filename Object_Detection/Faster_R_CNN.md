

## Faster R-CNN

Faster R-CNN được thiết kế để giải quyết vấn đề lớn nhất của Fast R-CNN: **sự chậm trễ do mô-đun đề xuất vùng (Region Proposal)**.

### I. Vấn đề cốt lõi của Fast R-CNN

1.  **Tốc độ chậm:** Fast R-CNN mất khoảng 2,3 giây để phát hiện vật thể trong một hình ảnh.
2.  **Thủ phạm:** Hầu hết thời gian này bị tiêu tốn bởi mô-đun **Tìm kiếm chọn lọc (Selective Search)**. Nếu bỏ qua thời gian của Selective Search, các mô-đun còn lại chỉ mất 0,32 giây.
3.  **Kết luận:** Selective Search là khâu chậm.

### II. Giải pháp: Mạng đề xuất vùng (RPN)

Để giải quyết vấn đề tốc độ, Faster R-CNN loại bỏ hoàn toàn Selective Search và thay thế bằng một mô-đun mạng CNN gọi là **RPN (Region Proposal Network)**. RPN thực hiện chức năng tương tự như Selective Search.

### III. Cơ chế hoạt động của RPN

Trước khi vào RPN, ảnh đầu vào sẽ được thay đổi kích thước sao cho chiều nhỏ nhất là 600.

1.  **Trích xuất đặc trưng:** Ảnh được đưa qua các tầng Max Pool, làm giảm kích thước không gian (ví dụ: từ 600x600 xuống 37x37). Điều này giống như việc chia ảnh đầu vào thành cấu trúc lưới 37x37.
2.  **Xử lý trong RPN:**
    *   Dữ liệu đi qua một tầng tích chập (convolutional layer) 3x3 để tăng trường tiếp nhận (receptive field).
    *   Sau đó, nó đi qua các tầng tích chập 1x1.

3.  **Sử dụng Anchor Box:** RPN sử dụng các **Hộp neo (Anchor Box)** là các cửa sổ có kích thước cố định, được đặt ở mọi vị trí trên cấu trúc lưới.
    *   Trong thực tế, người ta thường dùng **9 Hộp neo** (K=9) với các kích thước và tỷ lệ khung hình (aspect ratios) khác nhau tại mỗi vị trí.

4.  **Đầu ra của RPN:** RPN tạo ra hai loại đầu ra bằng cách sử dụng các tầng tích chập 1x1:

    *   **Phân loại (Classification):** Một tầng có $2K$ kênh (ví dụ: 18 kênh nếu K=9). Hai kênh này quyết định xem liệu có vật thể bên trong Hộp neo hay không. Kết quả này được đưa qua hàm SoftMax.
        *   Kênh thứ nhất cho biết xác suất **là vật thể**.
        *   Kênh thứ hai là ngược lại.
    *   **Hồi quy (Regression):** Một tầng khác có $4K$ kênh (ví dụ: 36 kênh nếu K=9). Bốn kênh này dùng để **điều chỉnh** Hộp neo (x, y, chiều rộng, chiều cao) sao cho nó bao phủ vật thể thực tế tốt nhất.

### IV. Hàm mất mát (Loss Function) cho RPN

Hàm mất mát của RPN là một **hàm mất mát đa nhiệm (multi-task loss)**, bao gồm hai thành phần:

1.  **Mất mát Phân loại (Classification Loss):** Tính toán xem Anchor Box có chứa vật thể hay không ($p_i$ là đầu ra của RPN; $p_i^*$ là Ground Truth).
2.  **Mất mát Hồi quy (Regression Loss):** Tính toán mức độ cần điều chỉnh Anchor Box ($t_i$ là đầu ra của mô hình; $t_i^*$ là Ground Truth).

#### Điều kiện áp dụng Regression Loss

Mất mát Hồi quy chỉ được áp dụng **nếu có vật thể trong vùng đó**. Điều này được kiểm soát bởi $p_i^*$ (Ground Truth):

*   $p_i^* = 1$ (Có vật thể, áp dụng hồi quy) nếu **IoU** (Tỷ lệ Giao thoa trên Hợp nhất) giữa Anchor Box và vật thể thực tế **lớn hơn 0.7**.
*   $p_i^* = 0$ (Không phải vật thể, bỏ qua hồi quy) nếu **IoU nhỏ hơn 0.3**.
*   Các Anchor Box có IoU nằm trong khoảng **từ 0.3 đến 0.7 sẽ bị bỏ qua** trong quá trình tính toán mất mát.
*   **Trường hợp đặc biệt:** Nếu không có Hộp neo nào đạt IoU > 0.7, các tác giả đề xuất sử dụng Hộp neo có **IoU cao nhất** làm dương tính (true positives) để đảm bảo mô hình có thể hội tụ tốt hơn.

### V. Thử thách khi huấn luyện (Imbalanced Data)

RPN có thể tạo ra số lượng Anchor Box rất lớn (ví dụ: 2400 vị trí * 9 neo = khoảng 20.000 neo).

1.  **Dữ liệu mất cân bằng:** Với số lượng neo lớn như vậy, mô hình thường có rất nhiều dự đoán sai và chỉ một vài dự đoán đúng, tạo ra vấn đề dữ liệu mất cân bằng (imbalanced data).
2.  **Giải pháp lấy mẫu:** Để khắc phục, người ta chỉ lấy mẫu một tập hợp con cân bằng để huấn luyện.
    *   Kích thước lô (batch size) thường là **256**.
    *   Mục tiêu là có **128 dự đoán dương tính** (positive) và **128 dự đoán âm tính** (negative).
    *   Nếu số lượng dương tính ít hơn 128 (ví dụ: chỉ 70), số lượng mẫu âm tính sẽ được tăng lên.

### VI. Quá trình huấn luyện đa bước

Việc huấn luyện Faster R-CNN rất khó khăn vì RPN (đề xuất vùng) phải được học, trong khi Selective Search trong Fast R-CNN là một thuật toán cố định. Tác giả đề xuất quy trình huấn luyện theo **nhiều bước**:

1.  **Bước 1: Huấn luyện RPN:** Khởi tạo mạng xương sống (backbone VGG) bằng trọng số Imagenet, sau đó huấn luyện mô-đun RPN. Điều này làm thay đổi trọng số mạng xương sống.
2.  **Bước 2: Huấn luyện Fast R-CNN:** Bỏ qua trọng số RPN đã thay đổi ở Bước 1. Khởi tạo mạng xương sống của Fast R-CNN bằng trọng số Imagenet. Trong bước này, RPN đã được huấn luyện (ở Bước 1) hoạt động như một mạng bên ngoài, **trọng số của nó không được cập nhật**.
3.  **Bước 3: Tinh chỉnh (fine-tune) RPN:** Huấn luyện RPN lại lần nữa để đề xuất vùng tốt hơn. Trong bước này, **trọng số mạng xương sống được đóng băng (freeze)**, chỉ huấn luyện mô-đun RPN.
4.  **Bước 4 (Cuối cùng):** Đóng băng cả mạng xương sống và mô-đun RPN, và **chỉ huấn luyện các tầng kết nối đầy đủ (fully connected layers)**.
