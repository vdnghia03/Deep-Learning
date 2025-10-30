
## Ví dụ Ma trận Trực quan về LoRA

Trong LoRA, chúng ta muốn cập nhật một Ma trận Trọng số gốc $W_0$ bằng cách thêm vào Ma trận Cập nhật $\Delta W$. Thay vì lưu trữ toàn bộ $\Delta W$, chúng ta sử dụng phương pháp phân tích ma trận (Matrix Decomposition) để biểu diễn nó bằng hai ma trận hạng thấp $W_B$ và $W_A$.

### 1. Thiết lập Ma trận (Giảm Bộ nhớ/Tham số)

Chúng ta hãy lấy ví dụ minh họa được đề cập trong nguồn tài liệu, sử dụng ma trận $100 \times 100$ và hạng thấp $R=5$.

#### A. Tinh chỉnh Toàn bộ (Full Fine-Tuning)

Nếu chúng ta tinh chỉnh toàn bộ ma trận cập nhật $\Delta W$ (kích thước $D \times K$), nơi $D=100$ và $K=100$:

*   **Ma trận Cập nhật ($\Delta W$):** Kích thước $100 \times 100$.
*   **Số lượng Tham số cần lưu trữ:** $100 \times 100 = 10.000$ phần tử.

#### B. Áp dụng LoRA (Low-Rank Adaptation)

LoRA phân tích $\Delta W$ thành tích của hai ma trận hạng thấp $W_B$ và $W_A$, với hạng $R=5$.

$$\Delta W \approx W_B W_A$$

1.  **Ma trận $W_A$ (biến đổi đầu vào):** Kích thước $R \times K$.
    *   $W_A$ có kích thước $5 \times 100$.
    *   Số tham số: $5 \times 100 = 500$ phần tử.
2.  **Ma trận $W_B$ (biến đổi đầu ra):** Kích thước $D \times R$.
    *   $W_B$ có kích thước $100 \times 5$.
    *   Số tham số: $100 \times 5 = 500$ phần tử.

| | Kích thước | Số lượng Tham số (Phần tử) |
| :--- | :--- | :--- |
| **Tinh chỉnh toàn bộ ($\Delta W$)** | $100 \times 100$ | $10.000$ |
| **LoRA ($W_B$ và $W_A$)** | $W_B (100 \times 5)$ và $W_A (5 \times 100)$ | $500 + 500 = 1.000$ |

**Kết quả:**
LoRA chỉ cần lưu trữ 1.000 tham số thay vì 10.000 tham số. Điều này tương đương với **giảm 10 lần** số lượng tham số cần huấn luyện và lưu trữ cho bản cập nhật.

Đây chính là cơ chế toán học giúp LoRA đạt được sự **giảm bộ nhớ khổng lồ** (ví dụ: giảm từ 1 TB xuống 25 MB trên GPT-3).

### 2. Giải thích về Hạng Nội tại Thấp (Low Intrinsic Rank)

Sự thành công của việc sử dụng $R=5$ (thay vì $R=100$) được hỗ trợ bởi giả thuyết Hạng Nội tại Thấp:

1.  **Mô hình có tính dư thừa:** Các mô hình tiền huấn luyện lớn có **chiều nội tại rất thấp**. Điều này có nghĩa là chúng có rất nhiều sự dư thừa (redundancy).
2.  **Hạng Ma trận:** Hạng của ma trận không nhất thiết bằng số chiều của nó. Trong ma trận $100 \times 100$, hạng tối đa là 100, nhưng hạng nội tại thực tế có thể chỉ là 70, 4, hoặc thậm chí thấp hơn.
3.  **Giả thuyết LoRA:** Nhóm LoRA giả thuyết rằng **bản cập nhật trọng số ($\Delta W$) cũng có hạng nội tại thấp**.
4.  **Hạng nhân tạo (Hyperparameter R):** Bằng cách chọn $R=5$, chúng ta **giới hạn hạng tối đa** của ma trận cập nhật $\Delta W$ thành 5. Giới hạn này được cho phép vì ta tin rằng ma trận có khả năng biểu đạt (express the information as richly) bằng ma trận nhỏ hơn nhiều.

### 3. Tăng Tốc độ Suy luận (Inference Speed)

Về mặt tính toán, ma trận hạng thấp còn giúp tăng tốc độ.

Trong quá trình suy luận, $W_0$ (trọng số gốc) được kết hợp với $W_B$ và $W_A$. Quá trình này được tối ưu hóa:

*   **Hợp nhất Trọng số:** Thay vì thực hiện phép nhân $W_B W_A$ mỗi khi thực hiện chuyển tiếp (forward pass), chúng ta **hợp nhất (merge) các bản cập nhật** này vào trọng số gốc $W_0$ trước khi suy luận.
    $$W_{\text{merged}} = W_0 + W_B W_A$$
*   **Kết quả:** Sau khi hợp nhất, mô hình suy luận trên $W_{\text{merged}}$ hoạt động như một mô hình truyền thống mà không có thêm bất kỳ ma trận phụ trợ nào. Điều này mang lại **độ trễ suy luận tiềm năng bằng không** (potentially zero inference latency), vốn là một lợi thế lớn của LoRA so với các phương pháp khác.
