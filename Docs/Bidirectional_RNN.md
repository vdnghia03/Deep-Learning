

## Bidirectional RNN (Bi-RNN)

Bi-RNN là kiến trúc mạnh mẽ trong xử lý mô hình trình tự (sequence model) vì nó tận dụng ngữ cảnh từ **toàn bộ** câu (trước và sau).

### 1. Hạn Chế của RNN Tiêu Chuẩn

**Ý tưởng:** Các mạng RNN thông thường (như Simple RNN, GRU, hoặc LSTM) chỉ sử dụng thông tin từ các từ đã xuất hiện **trước đó** để đưa ra dự đoán.

**Ví dụ Cụ thể (Nhận dạng Thực thể có Tên - NER):**

Xét từ "April" trong hai câu sau:

1.  "people think **April** is the most Pleasant **month** in India".
    *   *Dự đoán mong muốn:* April là **tháng**.
2.  "people think **April** is the **winner** of the tournament".
    *   *Dự đoán mong muốn:* April là **người/tên riêng**.

**Vấn đề:** Để phân biệt ý nghĩa của "April", mô hình cần ngữ cảnh từ các từ **theo sau** nó ("month" hay "winner"). Tuy nhiên, trạng thái ẩn $A$ của RNN tiêu chuẩn chỉ giữ ngữ cảnh của các từ **đã xuất hiện trước đó**.

### 2. Kiến Trúc Bi-RNN (Hai Mạng Ngược Chiều)

**Ý tưởng:** Bi-RNN khắc phục hạn chế trên bằng cách sử dụng hai mạng RNN chạy ngược chiều nhau để thu thập ngữ cảnh toàn diện.

**Ví dụ Cụ thể:**

| Thành phần | Hướng Di chuyển | Mục đích |
| :--- | :--- | :--- |
| **RNN Tiến (Forward RNN - Màu xanh lá)** | Từ đầu câu đến cuối câu | Thu thập ngữ cảnh **trước** (A\_forward). |
| **RNN Lùi (Backward RNN - Màu xanh dương)** | Từ cuối câu đến đầu câu | Thu thập ngữ cảnh **sau** (A\_backward). |

**Lưu ý:** Đây chỉ là **hai khối RNN** (tiến và lùi) hoạt động qua các mốc thời gian khác nhau, chứ không phải nhiều khối độc lập. Các khối này có thể là Simple RNN, **GRU** hoặc **LSTM**.

### 3. Thu Thập Ngữ Cảnh (A\_forward và A\_backward)

**Ý tưởng:** Tại mỗi mốc thời gian, trạng thái ẩn được tính toán từ cả hai hướng, đảm bảo ngữ cảnh của từ đó trong toàn bộ câu.

**Ví dụ Cụ thể (Tại vị trí $t=3$):**

Giả sử chúng ta muốn dự đoán cho từ $X_3$:

*   **$A_{forward 3}$ (Xanh lá):** Chứa thông tin ngữ cảnh của **tất cả các từ đã thấy trước đó** (tức là $X_1, X_2$).
*   **$A_{backward 3}$ (Xanh dương):** Chứa thông tin ngữ cảnh cho **tất cả các từ theo sau** điểm đó (tức là $X_4, X_5, ...$).

Quá trình này diễn ra tuần tự: RNN Tiến hoàn thành trước, sau đó RNN Lùi bắt đầu từ từ cuối cùng.

### 4. Cơ Chế Dự Đoán (Y)

**Ý tưởng:** Mô hình đưa ra dự đoán bằng cách sử dụng trạng thái ẩn đã được **ghép nối** (concatenated) từ hai hướng.

**Ví dụ Cụ thể (Công thức):**

Trong RNN tiêu chuẩn, dự đoán $Y$ chỉ dựa trên một vector ngữ cảnh $A$ (ngữ cảnh trước): $Y = Activation(W A + B)$.

Trong Bi-RNN, dự đoán $Y$ sử dụng vector ngữ cảnh ghép nối, cho phép $Y$ biết về ngữ cảnh của toàn bộ câu:

$$\mathbf{Y} = Activation(W \times [\mathbf{A_{forward}} \text{ concatenated } \mathbf{A_{backward}}] + B)$$

*   **$[\mathbf{A_{forward}} \text{ concatenated } \mathbf{A_{backward}}]$:** Đây là ma trận ghép nối hai vector ngữ cảnh. Ma trận này chứa thông tin ngữ cảnh đầy đủ (trước và sau) cho vị trí đang xét.
*   **W:** Ma trận trọng số được nhân với ma trận ghép nối.

### 5. Lan Truyền Ngược (Backpropagation)

**Ý tưởng:** Cơ chế Lan truyền Ngược (Backpropagation) hoạt động ngược chiều với quá trình tính toán ngữ cảnh.

**Ví dụ Cụ thể (Hướng Lan truyền):**

*   **RNN Tiến (Xanh lá):** Backpropagation di chuyển từ thời điểm cuối ($T_{cuối}$) về $T=1$.
*   **RNN Lùi (Xanh dương):** Backpropagation di chuyển từ thời điểm $T=1$ đến thời điểm cuối ($T_{cuối}$).

Cơ chế này đảm bảo rằng các tham số (weights) của cả hai mạng được cập nhật chính xác dựa trên lỗi dự đoán.
