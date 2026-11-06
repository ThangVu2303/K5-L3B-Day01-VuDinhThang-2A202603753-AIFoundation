# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603753
- **Họ và tên:** Vũ Đình Thắng
- **SEED:** 7072

---

## 1. Bằng chứng

| Tệp | Có? |
| --- | :---: |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

---

## 2. Kết quả chính

Bảng so sánh 4 thế hệ AI trên cùng bài toán lọc tin nhắn SMS Spam (trích xuất trực tiếp từ file `lab1_report.json` của tác giả):

| Hàng / Mô hình | Precision | Recall | F1 (test) | F1 (SUB 40) | Train (s) | Tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. BoW + LogReg (tự code)** | 0.851 | 0.989 | **0.915** | 0.974 | 4.9s | 7,701 | 0.06 ms | 0.13 ms | $0.000 |
| **2. BoW + MLP (deep learning)** | 0.861 | 0.978 | **0.916** | 0.974 | 6.4s | 35,348 | 0.33 ms | 0.46 ms | $0.000 |
| **3a. Encoder (MiniLM) + kNN** | 0.941 | 0.841 | **0.888** | 1.000 | 86.2s | 20.8 | 60.39 ms | 88.59 ms | $0.000 |
| **3b. Encoder (MiniLM) + LogReg** | 0.921 | 0.989 | **0.954** | 1.000 | 86.9s | 30.7 | 36.25 ms | 53.73 ms | $0.000 |
| **3c. Encoder (MiniLM) + MLP head** | 0.911 | 0.989 | **0.948** | 1.000 | 87.4s | 31.2 | 32.36 ms | 50.36 ms | $0.000 |

*Ghi chú:*
- Hàng LLM chạy zero-shot trên tập SUB 40 tin đạt F1 dao động từ 0.90 – 0.95, tuy nhiên độ trễ mạng p95 cao (500 – 1200 ms/tin) và tốn chi phí API liên tục.
- Mô hình **3b. Encoder + LogReg** đạt F1 tổng thể cao nhất trên tập test thật (F1 = 0.954) với độ trễ thấp và chi phí 0 USD.

---

## 3. Dự đoán và thực tế

Ba dự đoán lệch nhiều nhất so với kết quả thực nghiệm:

1. **Dự đoán `dl_curve` (Thí nghiệm 2.A — MLP so với Logistic Regression):**
   - *Dự đoán:* Tôi từng đoán rằng khi thêm lớp ẩn 64 nơ-ron và hàm kích hoạt phi tuyến ReLU, mô hình Deep Learning (MLP) sẽ đạt điểm F1 vượt trội so với Logistic Regression đơn giản ở Phần 1 nhờ khả năng học các tổ hợp từ khóa.
   - *Thực tế:* F1 của MLP là **0.916**, gần như bằng hệt F1 của Logistic Regression (**0.915**) (chênh lệch chỉ 0.001).
   - *Vì sao lệch:* Đầu vào Bag-of-Words là các vector nhị phân rời rạc và tập dữ liệu SMS khá nhỏ (~3.500 mẫu train). Mối quan hệ giữa các từ khóa spam ("free", "claim", "prize") với nhãn vốn dĩ đã mang tính phân tách tuyến tính rất rõ nét. Việc thêm tầng ẩn làm tăng số tham số từ 2.001 lên 128.129 không trích xuất thêm được nhiều ngữ nghĩa mà trái lại còn khiến mô hình nhanh chóng bị quá khớp (overfitting) sau epoch 8.

2. **Dự đoán `tradeoff` (Độ trễ của kNN so với LogReg trên không gian Embedding):**
   - *Dự đoán:* kNN không mất thời gian huấn luyện trọng số nên sẽ có tốc độ suy luận nhanh và độ trễ cạnh tranh tốt với LogReg.
   - *Thực tế:* kNN có độ trễ suy luận rất cao ($p95 = 88.59$ ms), cao hơn LogReg ($53.73$ ms) và vượt ra ngoài ngưỡng SLA 50 ms của hệ thống viễn thông. Tốc độ cả lô của kNN chỉ đạt ~20.8 tin/s.
   - *Vì sao lệch:* kNN là thuật toán "học lười" (lazy learning). Giai đoạn train thực chất chỉ là ghi nhớ dữ liệu vào RAM. Đến khi dự đoán từng tin nhắn, nó bắt buộc phải tính khoảng cách Cosine với toàn bộ hơn 3.600 vector mẫu trong tập train ($O(N)$), khiến độ trễ dự đoán từng tin bị kéo dài. Ngược lại, LogReg đã học sẵn ma trận trọng số gọn nhẹ nên chỉ tốn 1 phép nhân ma trận $1 \times 384$ là xong.

3. **Dự đoán `hard_eval` (Khả năng chống chịu của BoW trước 50 case bẫy đối kháng):**
   - *Dự đoán:* Tôi đoán mô hình BoW vẫn sẽ giữ được độ chính xác ở mức chấp nhận được (~60% – 65%) trên bộ 50 case khó.
   - *Thực tế:* BoW bị tụt rất nặng xuống dưới 50% (đoán sai phần lớn nhóm tin ham chứa từ khóa và nhóm né từ khóa).
   - *Vì sao lệch:* BoW hoàn toàn phụ thuộc vào sự xuất hiện của từ đơn lẻ trong từ điển cố định. Khi người tạo cố tình viết câu bạn bè nhưng chèn từ "free pizza", "won the bet", "claim seat", BoW bị đánh lừa 100% vì các từ này có trọng số spam rất lớn, hoàn toàn mù tịt trước ngữ cảnh đùa vui hay tình huống giao tiếp đời thường.

---

## 4. Phân tích 50 case khó

- **Model tụt nhiều nhất:** Mô hình **BoW (Phần 1 & 2)** bị tụt thảm hại nhất, từ F1 test thường ~0.915 tụt dốc xuống chỉ còn ~0.50 – 0.55 trên bộ 50 case khó.
- **Tụt mạnh nhất ở 2 nhóm case:**
  1. *Tưởng spam nhưng không phải (`ham_looks_spam`):* 10 tin nhắn bình thường của bạn bè, gia đình nhưng cố tình chứa các từ khóa kích thích như "FREE", "WON", "URGENT", "Claim".
  2. *Né từ khoá (`obfuscation`):* 6 tin nhắn spam lừa đảo cố tình viết chèn ký tự lạ, số thay chữ (`F R E E`, `c.a.s.h`, `C0ngr4ts`).

- **Hai case cụ thể và cơ chế khiến model sai:**
  1. **Case #1 (`ham_looks_spam`):**
     - *Nội dung:* `"Congrats!!! You WON the bet lol, I owe you a beer. Call me later"` (Nhãn thật: `ham`).
     - *Cơ chế sai của BoW:* Tin nhắn chứa các từ khóa spam kinh điển: `"Congrats"`, `"WON"`, `"Call"`. Trong mô hình BoW, các từ này có trọng số dương rất lớn kéo xác suất về spam. Vì BoW chỉ cộng điểm từng từ độc lập mà không hiểu cấu trúc ngữ cảnh bạn bè cá cược đùa vui (`"I owe you a beer"`), mô hình tự tin phán đoán là **SPAM** (sai).
  2. **Case #31 (`obfuscation`):**
     - *Nội dung:* `"C0ngr4tulat10ns! Y0u have w0n a £1000 c.a.s.h pr1ze. Cl41m n0w at bit.ly/claim"` (Nhãn thật: `spam`).
     - *Cơ chế sai của BoW:* Kẻ lừa đảo sử dụng kỹ thuật Leetspeak (thay chữ cái bằng số `0`, `4`, `1` và chèn dấu chấm `c.a.s.h`). Khi đưa qua hàm tiền xử lý `tokenize`, các từ dị dạng này không khớp với bất kỳ từ nào trong từ điển 2.000 từ của BoW. Vector đầu vào toàn số 0, mô hình hoàn toàn mù trước tín hiệu lừa đảo và dự đoán là **HAM** (sai).

---

## 5. Thử thách cá nhân

- **Đề bài theo SEED 7072 (Đề #5):**
  > *"Embedding · Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin."*

- **Cách làm:**
  Viết 4 tin nhắn kiểm thử trực tiếp trong notebook ở Cell 96:
  1. *Ham 1 (chứa 'free'):* `"Hey are you free this Sunday? We want to take you out for birthday dinner"` (Nhãn: 0 - ham).
  2. *Ham 2 (chứa 'win'):* `"Did our school debate team win the championship game yesterday?"` (Nhãn: 0 - ham).
  3. *Spam 1 (né từ khóa):* `"Exclusive luxury getaway reserved for your mobile. Reply to reserve your spot"` (Nhãn: 1 - spam).
  4. *Spam 2 (né từ khóa):* `"Security alert: your account rewards expire at midnight, verify details to claim"` (Nhãn: 1 - spam).

- **Kết quả thực nghiệm:**
  - **BoW + LogReg:** Đúng 1/4 (Đoán sai cả 2 tin ham thành spam do dính từ 'free' và 'win'; bỏ lọt 1 tin spam né từ).
  - **Encoder (MiniLM) + LogReg:** Đoán đúng **4/4 (100%)**.

- **Giải thích cơ chế:**
  - *Với 2 tin ham:* BoW chỉ đếm từ độc lập, thấy 'free' và 'win' là cộng điểm spam rất nặng nên bị lừa. Embedding (MiniLM) mã hóa trọn vẹn ngữ cảnh câu hỏi hẹn đi ăn sinh nhật và hỏi thăm giải thi đấu, đưa vector vào đúng cụm trò chuyện bạn bè thân mật (ham).
  - *Với 2 tin spam:* Kẻ lừa đảo né các từ 'free', 'prize', 'cash' quen thuộc nhưng vẫn giữ ngữ nghĩa tiếp thị, câu kéo ('exclusive getaway', 'rewards expire'). Không gian vector của MiniLM đặt các câu này gần vùng của tin nhắn quảng cáo/lừa đảo, giúp bộ phân loại tuyến tính nhận diện chính xác là spam.

---

## 6. Phương án cho sản phẩm thật

- **Yêu cầu bài toán:** Nhà mạng viễn thông cần lọc 10 triệu SMS/ngày (lưu lượng trung bình ~116 tin/giây), yêu cầu nghiêm ngặt độ trễ $p95 < 50$ ms/tin.
- **Phương án lựa chọn:** Chọn **Mô hình 3b. Encoder (MiniLM) + LogReg**.

- **Dẫn chứng số liệu từ bài làm:**
  1. *Về chất lượng:* Đạt F1 cao nhất trên tập test thực tế (**0.954** so với 0.915 của BoW), khả năng kháng cự vượt trội trước các tin nhắn ngụy trang né từ khóa.
  2. *Về độ trễ:* Đo trên CPU Colab đạt $p95 = 53.7$ ms. Khi triển khai trên môi trường production với runtime tối ưu (như ONNX Runtime / TensorRT hoặc GPU inference), độ trễ $p95$ dễ dàng đạt dưới **15 – 20 ms**, hoàn toàn thỏa mãn SLA 50 ms.
  3. *Về thông lượng:* Một worker xử lý được ~31 tin/s. Chỉ cần triển khai cụm 4–8 worker song song là đáp ứng thừa sức lưu lượng 116 tin/s của nhà mạng.
  4. *Về chi phí:* Chi phí API là **0.000 USD/1k tin** (chạy on-premise nội bộ, chỉ tốn tiền thuê server khoảng 50 – 100 USD/tháng).

- **So sánh và bác bỏ các phương án khác:**
  - *Bác bỏ LLM:* Độ trễ quá lớn ($p95 > 500 – 1200$ ms, vượt ngưỡng SLA từ 10 đến 25 lần). Về chi phí: Với 10 triệu tin/ngày (300 triệu tin/tháng), dùng model rẻ nhất `gpt-6-luna` tốn **$3{,}150 USD/tháng** (~80 triệu VNĐ), model lớn tốn hơn **$31{,}500 USD/tháng** (~800 triệu VNĐ), chi phí bất khả thi đối với bài toán lọc SMS.
  - *Bác bỏ kNN (3a):* Độ trễ suy luận quá lớn ($p95 = 88.6$ ms) do phải so sánh với hàng triệu vector mẫu trong bộ nhớ, không thể mở rộng (scale) khi tập dữ liệu phình to.
  - *Bác bỏ BoW (1 & 2):* Dù độ trễ siêu nhanh (< 1 ms) nhưng chất lượng lọc kém trước các bẫy né từ và nhận nhầm quá nhiều tin nhắn giao dịch/bạn bè hợp lệ.

- **Đánh đổi chấp nhận:**
  - Chấp nhận tốn thời gian mã hóa trước tập train (~86 giây) và tốn tài nguyên RAM/VRAM để tải mô hình MiniLM (~90MB bộ nhớ) thay vì mô hình siêu nhẹ chỉ vài KB như BoW.

---

## 7. AI log và điều đã học

### Bảng ghi chép nhật ký sử dụng AI (AI Log)

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời / viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| **Phần 2** | Cách tính số tham số của `build_mlp(2000, 64)` và tại sao val loss lại tăng sau vài epoch? | Hướng dẫn công thức `(d_in + 1) * d_out` và giải thích hiện tượng Overfitting khi mạng học vẹt dữ liệu train. | Tự tính nhẩm lại trên giấy: Lớp 1 có 128.064 tham số, Lớp 2 có 65 tham số, tổng 128.129; kiểm tra khớp hàm `sum(p.numel() ...)` trong PyTorch. |
| **Phần 3** | Viết hàm cosine similarity và top k láng giềng bằng NumPy không dùng vòng lặp qua từng dòng? | Đề xuất chuẩn hóa L2 norm trước rồi dùng phép nhân ma trận `M @ q` và `np.argsort(scores)[::-1][:k]`. | Kiểm tra xử lý trường hợp chia cho 0 (`eps = 1e-12`), đảo chiều sắp xếp giảm dần và chạy qua 2 ô test kiểm tra ở Cell 48 và 53. |
| **Phần 4** | Cách viết regex cho hàm parse_label để không bị nhầm khi LLM trả lời "Not spam"? | Gợi ý bóc tách các token chữ cái và kiểm tra điều kiện "not" đi kèm "spam" trước khi kiểm tra "spam". | Thử nghiệm với các chuỗi thử thách: 'Spam.', 'Label: spam', 'Not spam', 'I am not sure' để đảm bảo trả về đúng 0, 1 hoặc None. |
| **Phần 5** | Đề xuất thiết kế hệ thống Router kết hợp giữa Embedding và LLM để vừa nhanh vừa tiết kiệm tiền? | Đề xuất kiến trúc 2 tầng: dùng xác suất của LogReg làm cổng lọc sơ bộ, chỉ gửi tin nằm trong khoảng phân vân sang LLM. | Tính toán chi phí thực tế: giảm được 95% chi phí API hàng tháng mà vẫn đảm bảo độ trễ $p95 < 50$ ms cho phần lớn tin nhắn. |

### Điều bạn hiểu ra hôm nay mà trước đó chưa hiểu

1. **Hiểu lầm về Deep Learning:** Trước đây tôi nghĩ cứ dùng mạng nơ-ron sâu (Deep Learning) với nhiều tham số thì kết quả sẽ luôn vượt trội Machine Learning truyền thống. Qua bài lab này, tôi hiểu rằng với dữ liệu dạng túi từ (Bag-of-Words) trên tập mẫu nhỏ, mô hình ML cổ điển (như Logistic Regression) vừa chạy nhanh hơn gấp hàng chục lần, vừa ít bị quá khớp (overfitting) và cho kết quả tương đương.
2. **Vai trò sống còn của Biểu diễn đặc trưng (Representation):** Chìa khóa tạo nên sự đột phá trong NLP không nằm ở độ sâu của mạng phân loại bên trên, mà nằm ở **cách biến văn bản thành vector**. Việc chuyển đổi từ Bag-of-Words (vector đếm từ rời rạc) sang Embedding (vector tọa độ ngữ nghĩa liên tục) mới là bước tiến thực sự giúp mô hình phân biệt được ngữ cảnh và chống lại các chiêu trò ngụy trang né từ khóa.
