# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603639
- **Họ và tên:** Bùi Quang Hiếu
- **SEED:** 6046

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook có các output thí nghiệm | ☑ |
| [`lab1_report.json`](lab1_report.json) — kết quả từ ô "Báo cáo cuối" | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

`lab1_report.json` có 15/15 kiểm tra đạt và 10 dự đoán. Bản notebook đính kèm có output của các thí nghiệm, nhưng ô "Báo cáo cuối" trong bản đó chưa lưu output; file JSON đã được tạo riêng ở `src/` rồi sao chép vào thư mục nộp.

## 2. Kết quả chính

Số liệu từ `lab1_report.json` (độ trễ và tốc độ phụ thuộc lần chạy; LLM chỉ được đánh giá trên SUB 40):

| Phương án | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg | 0,943 | 0,947 | 0,032 | 0 |
| 2. BoW + MLP | 0,965 | 0,974 | 0,118 | 0 |
| 3a. MiniLM + kNN | 0,925 | 0,950 | 116,282 | 0 |
| 3b. MiniLM + LogReg | 0,956 | 1,000 | 7,741 | 0 |
| 3c. MiniLM + MLP head | 0,971 | 1,000 | 11,683 | 0 |
| 4a. LLM nhỏ zero-shot | — | 0,923 | 1.887,482 | 0,00566 |
| 4b. LLM lớn zero-shot | — | 0,974 | 1.969,295 | 0,05660 |
| 4c. LLM nhỏ few-shot | — | 0,974 | 1.717,957 | 0,01646 |

Chỉ so sánh trực tiếp chất lượng với LLM bằng cột **F1 SUB 40**. F1 bằng 1,000 trên 40 tin không chứng minh mô hình không mắc lỗi trên tin mới.

## 3. Dự đoán và thực tế

1. **`ml_f1`:** Tôi dự đoán BoW + LogReg có F1 khoảng 0,80–0,85; kết quả test là **0,943**, precision **0,989** và recall **0,901**. Precision cao hơn recall đúng như dự đoán nhưng F1 cao hơn nhiều: từ điển BoW vẫn nắm được nhiều mẫu từ vựng quen thuộc của SMS trong tập test này.
2. **`dl_curve`:** Tôi dự đoán MLP có F1 xấp xỉ hoặc thấp hơn LogReg vì có nhiều tham số và có thể overfit. Thực tế MLP đạt **0,965** so với **0,943** của BoW + LogReg, cao hơn **0,022**; val loss thấp nhất ở chỉ số epoch 7. Dự đoán của tôi đã đánh giá thấp khả năng hidden layer học tổ hợp từ, dù chênh lệch nhỏ vẫn cần thận trọng khi khái quát.
3. **`hard_eval`:** Tôi dự đoán accuracy trên 50 case khó của BoW khoảng 60–75% và LLM khoảng 75–90%, nhóm prompt injection sẽ khó nhất với LLM. Kết quả BoW + LogReg và BoW + MLP đều chỉ **50%**, còn LLM nhỏ zero-shot **98%**; LLM đúng **6/6** case prompt injection. Bộ case có chủ đích khác nhiều so với test thường: BoW hụt spam giọng tự nhiên, trong khi LLM lần này nhận ra lệnh chèn trong nội dung là dữ liệu cần phân loại.

## 4. Phân tích 50 case khó

**BoW + LogReg tụt nhiều nhất** khi so F1 SUB 40 với 50 case khó: **0,947 → 0,286** (giảm 0,661); BoW + MLP giảm **0,974 → 0,390**. BoW + LogReg đúng **0/10 spam giọng tự nhiên**, **1/8 tin thao túng** và **1/6 tin né từ khoá**. Tập test thường cùng kiểu dữ liệu train nên có thể cho điểm quá lạc quan đối với những tin cố tình đổi cách diễn đạt.

- **Case #11, spam giọng tự nhiên:** tin giả giọng người quen ở phòng tập mời tham gia ưu đãi thành viên, yêu cầu trả lời để đăng ký. BoW + LogReg xếp nhầm ham vì không thấy mẫu từ khoá spam quen thuộc; thông điệp thực chất vẫn là quảng cáo không yêu cầu.
- **Case #12, spam giọng tự nhiên:** tin mở đầu như lời làm quen rồi dụ người nhận xem hồ sơ trên một trang hẹn hò. BoW + LogReg xếp nhầm ham vì từng token trông giống hội thoại cá nhân, còn ý định dẫn sang trang quảng bá chỉ rõ khi đọc cả câu.

Trên cùng bộ 50 tin, 3a/3b/3c có F1 lần lượt **0,714/0,640/0,612**, trong khi LLM nhỏ zero-shot đạt **0,983** và prompt phòng thủ đạt **1,000**. Cả 4a và 4d đều đúng **6/6 prompt injection**; 4d sửa thêm một lỗi ở nhóm ham trông như spam, nhưng 50 case viết tay không đại diện cho phân phối SMS thật.

## 5. Thử thách cá nhân

**Đề #5 (SEED 6046):** viết 2 spam né từ khoá và 2 ham chứa “free”/“win”, so BoW với Embedding + LogReg. Tôi viết bốn câu tiếng Anh trong ô thử thách của notebook, tự đặt nhãn, dùng `vectorize` + `predict_proba` cho BoW và `emb_model.encode` + chuẩn hoá `(emb - mu) / sd` + `predict_proba` cho mô hình 3b, cùng ngưỡng 0,5.

| Tin tự viết (tóm tắt) | Nhãn thật | BoW | Embedding + LogReg | Giải thích |
| --- | --- | --- | --- | --- |
| “Your account requires immediate verification… avoid interruption.” | spam | ham (sai) | spam (đúng) | Mạo danh thông báo xác minh tài khoản, tạo áp lực; không dùng từ spam kiểu “free/prize”. |
| “We have an important update regarding your payment details… attached link.” | spam | ham (sai) | spam (đúng) | Dụ xác nhận thông tin thanh toán qua liên kết bằng giọng thông báo bình thường; BoW khó nhận ý định nếu chỉ nhìn từ rời rạc. |
| “Are you free this evening? We can meet at the library…” | ham | ham (đúng) | ham (đúng) | “free” nghĩa là rảnh, ngữ cảnh hẹn học ở thư viện không phải ưu đãi. |
| “I hope our team can win the football match tomorrow…” | ham | ham (đúng) | ham (đúng) | “win” nói về trận bóng, không phải thông báo trúng thưởng. |

Kết quả: **BoW đúng 2/4, Embedding + LogReg đúng 4/4** trên bốn tin này. Đó chỉ là bốn ví dụ tự viết, không phải ước lượng hiệu quả chung.

## 6. Phương án cho sản phẩm thật

**Chọn 3b MiniLM + LogReg** làm phương án ban đầu: p95 **7,741 ms/tin < 50 ms**, thông lượng đo cả lô khoảng **202 tin/giây** so với nhu cầu trung bình khoảng **116 tin/giây**. F1 toàn bộ test **0,956**, F1 SUB 40 **1,000**, F1 trên 50 case khó **0,640**; encoder đóng băng, thời gian encode + train head khoảng **11,338 giây** trong lần đo, không có phí API mỗi tin. 3c tốt hơn trên test thường (**0,971**) nhưng chậm hơn (**11,683 ms p95**) và thấp hơn trên 50 case khó (**0,612**); cần tập kiểm thử đại diện lớn hơn trước khi chốt phương án cuối. Thông lượng trung bình đo trong lab chưa bảo đảm tải cao điểm của 10 triệu SMS/ngày.

Đánh đổi: cần dữ liệu có nhãn để huấn luyện và cập nhật LogReg; model vẫn bỏ sót nhiều spam giọng tự nhiên (**1/10 đúng** trong nhóm này ở bộ case khó). Có thể định tuyến tin không chắc chắn sang hàng kiểm duyệt riêng, nhưng gọi LLM đồng bộ sẽ làm chính các tin đó vượt SLA 50 ms. Nếu gửi **toàn bộ** 10 triệu tin/ngày qua LLM nhỏ zero-shot, mức giá *ước tính* từ `lab1_report.json` là **$0,00566/1.000 tin × 10.000 × 30 ≈ $1.698/tháng**, p95 đo được **1.887 ms/tin**, không đạt SLA; phí và độ trễ thực tế có thể khác theo tải và độ dài tin.

## 7. AI log và điều đã học

| Phần | Đã hỏi AI | AI trả lời/viết | Tôi sửa hoặc kiểm chứng |
| --- | --- | --- | --- |
| Lý thuyết 2–3 | Giải thích overfit, gradient, embedding và cách đọc kết quả. | Gợi ý giải thích các bước train và so sánh BoW/embedding. | Đối chiếu F1, val loss và output 3.B trong notebook; giữ lại các câu trả lời phù hợp với số đo. |
| Phần 4 | Nhờ hoàn thành TODO 4.1–4.4. | Viết hàm gọi API, tạo prompt, đọc nhãn và phân loại lô. | Chạy kiểm tra API giả lập; khi API thật trả nhãn rỗng, xem `finish_reason='length'`, thử 16 token trên một tin rồi chạy lại để có `invalid=0`. |
| Phần 5 | Hỏi về Pareto, stress test, tính tay và cách điền báo cáo. | Đưa cách tính, giải thích giới hạn tập SUB 40 và so sánh các mô hình. | Tự viết phép tính trên giấy; kiểm tra 5/5 đáp án và đối chiếu các bảng đã lưu; tự viết bốn tin ở thử thách cá nhân và chạy hai mô hình. |

Trước bài lab, tôi tưởng F1 cao trên tập test thường có thể áp dụng cho mọi loại spam; 50 case khó cho thấy BoW rất dễ bỏ sót tin giả giọng người quen. Tôi cũng hiểu rằng một phản hồi LLM rỗng do giới hạn token phải được đếm là nhãn không hợp lệ, không thể coi F1=0 trong lần chạy lỗi là chất lượng thật của model. Tập SUB chỉ có 40 tin nên những chênh lệch F1 nhỏ cần được kiểm chứng trên nhiều dữ liệu hơn.
