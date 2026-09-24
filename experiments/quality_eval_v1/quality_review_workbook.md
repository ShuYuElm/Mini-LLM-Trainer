# 独立回答质量评估 · 评分手册

<!-- prompt_set_sha256: c6d0fce2304cc0fe576af3c78d4bcd266a27b9e1a1c2bfce468a35f55b0a06ec -->

- 题集哈希：`c6d0fce2304cc0fe576af3c78d4bcd266a27b9e1a1c2bfce468a35f55b0a06ec`
- 题目数：40；每题候选数：3
- 每题候选顺序独立随机，种子：20260921
- 完整题集：true

> 评分期间不要打开 `quality_review_mapping.json` 或 `quality_generation.json`：
> 前者直接给出标签到模型的映射，后者按模型标识保存回答，打开就会破坏盲评。

## 评分维度（每维 0/1/2）

| 维度 | 0 分 | 1 分 | 2 分 |
| --- | --- | --- | --- |
| 正确性 | Wrong main answer, contradiction, or unusable response. | Partly correct but includes a factual or reasoning error. | All stated task-relevant facts and reasoning are correct. |
| 完整性 | Misses the main requested information or action. | Includes the main information but misses at least one required fact. | Covers every required fact or action. |
| 格式遵循 | Substantially violates the required output form. | Mostly follows the form with a minor violation. | Satisfies every explicit output-format requirement. |
| 无无据添加 | Adds a material invented fact, new task, promise, or changed intent. | Adds minor unsupported or unnecessary content without changing the main answer. | Adds no unsupported facts or unrequested requirements. |

## 填写方式

每个候选回答下面都有一行待填，把四个下划线换成 0/1/2，并在 `|` 后写一句理由：

```text
A: 2 1 2 2 | 日期和地点正确，漏了报名截止日；单句无标题；无编造内容。
```

- 四个数字依次是：正确性、完整性、格式遵循、无无据添加。
- 只能填 0、1、2，理由不能为空。
- 参考答案只是示例，摘要/改写/推理接受等价表述；缺信息记完整性，说错记正确性。
- 同一题的三个候选各自独立打分，不要只排序。

填完在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe import_quality_scores.py
```

该命令只读本文件并写入 `quality_review_scores.json`，不会读取映射文件。

---

## 1. `sum_workshop`（摘要）

**Instruction**

```text
Write a single-sentence summary stating the revised workshop date, venue, and registration deadline.
```

**Input**

```text
The mapmaking workshop was planned for April 11 in Studio 2. It will now take place on April 18 in Studio 5. Registration closes on April 14. The fee remains unchanged.
```

**参考答案（示例，不是唯一正确答案）**

```text
The mapmaking workshop will be held on April 18 in Studio 5, with registration closing on April 14.
```

**必需信息**：April 18；Studio 5；Registration closes April 14

**格式要求**：Exactly one sentence, no heading.

**禁止添加**：No invented dates, locations, fees, or consequences.

### 候选 A

```text
The mapmaking workshop, originally scheduled for April 11 in Studio 2, has been moved to April 18 in Studio 5, with registration deadline set for April 14. The fee remains unchanged.
```

A: 2 2 1 1 | 日期、地点和报名截止日均正确，但用了两句，并额外保留了原日期、原场地和费用信息。

### 候选 B

```text
The mapmaking workshop will be held on April 18 in Studio 5, with registration closing on April 14, and the fee unchanged.
```

B: 2 2 2 1 | 三项必需信息均正确且为单句，但额外加入了“费用不变”的非必需信息。

### 候选 C

```text
The mapmaking workshop, originally scheduled for April 11 in Studio 2, has been rescheduled to April 18 in Studio 5, with registration deadline set for April 14. The fee has not been altered.
```

C: 2 2 1 1 | 三项必需信息均正确，但用了两句，并额外保留了原日期、原场地和费用信息。


## 2. `sum_deliveries`（摘要）

**Instruction**

```text
Summarize the delivery change in one sentence, including the affected area, closure period, and temporary pickup point.
```

**Input**

```text
Deliveries to Cedar Court are paused from June 3 through June 6 while the access road is resurfaced. During that period, residents can collect packages at the north depot. Deliveries elsewhere continue normally.
```

**参考答案（示例，不是唯一正确答案）**

```text
Cedar Court deliveries are paused June 3 through June 6, with packages available at the north depot.
```

**必需信息**：Cedar Court deliveries paused；June 3 through June 6；Pickup at north depot

**格式要求**：Exactly one sentence, no heading.

**禁止添加**：No invented dates, locations, fees, or consequences.

### 候选 A

```text
Deliveries to Cedar Court are temporarily suspended from June 3 to June 6, with the access road resurfacing, and residents can pick up packages at the north depot during this period.
```

A: 2 2 2 1 | 受影响区域、日期和取件点均正确且为单句，但额外说明了道路施工原因。

### 候选 B

```text
Deliveries to Cedar Court are temporarily suspended from June 3 to June 6, with residents able to pick up packages at the north depot during this period.
```

B: 2 2 2 2 | 受影响区域、暂停日期和北侧仓库取件点齐全，单句且无额外内容。

### 候选 C

```text
Deliveries to Cedar Court are temporarily halted from June 3 to June 6, with the access road resurfacing, and residents can pick up packages at the north depot during this period.
```

C: 2 2 2 1 | 受影响区域、日期和取件点均正确且为单句，但额外说明了道路施工原因。


## 3. `sum_membership`（摘要）

**Instruction**

```text
In one sentence, report the new monthly price, its start date, and which members are exempt.
```

**Input**

```text
Starting August 1, monthly membership will cost 28 dollars instead of 24 dollars. Members on annual plans keep their existing price until renewal. Opening hours will not change.
```

**参考答案（示例，不是唯一正确答案）**

```text
Monthly membership rises to 28 dollars on August 1, while annual-plan members keep their existing price until renewal.
```

**必需信息**：28 dollars per month；Starts August 1；Annual plans retain current price until renewal

**格式要求**：Exactly one sentence, no heading.

**禁止添加**：No invented dates, locations, fees, or consequences.

### 候选 A

```text
Starting August 1, monthly membership will cost 28 dollars, and members on annual plans will keep their existing price until renewal. Opening hours will not change.
```

A: 2 2 1 1 | 新月费、生效日和年付会员例外均正确，但输出两句且额外加入营业时间信息。

### 候选 B

```text
Starting August 1, monthly membership will cost 28 dollars, and members on annual plans will keep their existing price until renewal. Opening hours will not change.
```

B: 2 2 1 1 | 新月费、生效日和年付会员例外均正确，但输出两句且额外加入营业时间信息。

### 候选 C

```text
Starting August 1, monthly membership will cost 28 dollars, and members on annual plans will keep their existing price until renewal.
```

C: 2 2 2 2 | 28美元、8月1日生效及年付会员续费前维持原价均完整，且严格为单句。


## 4. `sum_water`（摘要）

**Instruction**

```text
Produce a two-sentence summary covering the affected buildings, interruption time, and drinking-water arrangement.
```

**Input**

```text
Water will be unavailable in Buildings F and H on May 9 from 10 a.m. to 1 p.m. Building G is unaffected. Free bottled drinking water will be available in the courtyard during the interruption.
```

**参考答案（示例，不是唯一正确答案）**

```text
Buildings F and H will have no water on May 9 from 10 a.m. to 1 p.m. Free bottled drinking water will be available in the courtyard.
```

**必需信息**：Buildings F and H；May 9, 10 a.m. to 1 p.m.；Free bottled drinking water in courtyard

**格式要求**：Exactly two sentences, no heading.

**禁止添加**：No invented dates, locations, fees, or consequences.

### 候选 A

```text
Water will be unavailable in Buildings F and H on May 9 from 10 a.m. to 1 p.m., while Building G is unaffected. Free bottled drinking water will be available in the courtyard during the interruption.
```

A: 2 2 2 1 | F、H楼、5月9日10点至13点及庭院免费瓶装水均正确，两句格式合规，但额外提到G楼不受影响。

### 候选 B

```text
Water will be unavailable in Buildings F and H on May 9 from 10 a.m. to 1 p.m., while Building G is unaffected. Free bottled drinking water will be available in the courtyard during the interruption.
```

B: 2 2 2 1 | F、H楼、5月9日10点至13点及庭院免费瓶装水均正确，两句格式合规，但额外提到G楼不受影响。

### 候选 C

```text
The affected buildings are F and H, with interruptions from 10 a.m. to 1 p.m., and free bottled drinking water will be available in the courtyard during this period.
```

C: 2 1 1 2 | 楼栋、时间段和饮水安排正确，但漏掉5月9日，且只写了一句而非两句。


## 5. `sum_bus`（摘要）

**Instruction**

```text
Summarize this notice in one sentence, preserving the route number, detour dates, and temporary stop.
```

**Input**

```text
Route 26 will bypass Harbor Street from September 12 through September 15 because of bridge inspections. Passengers should use the temporary stop outside the aquarium. The first and last bus times are unchanged.
```

**参考答案（示例，不是唯一正确答案）**

```text
Route 26 bypasses Harbor Street September 12 through September 15, with a temporary stop outside the aquarium.
```

**必需信息**：Route 26 bypasses Harbor Street；September 12 through September 15；Temporary stop outside aquarium

**格式要求**：Exactly one sentence, no heading.

**禁止添加**：No invented dates, locations, fees, or consequences.

### 候选 A

```text
Route 26 will be rerouted to bypass Harbor Street from September 12 to 15 due to bridge inspections, with temporary stops outside the aquarium and no changes to bus times.
```

A: 1 2 2 1 | 路线、日期和水族馆外临时站信息齐全，但把单个临时站写成复数，并额外加入检查原因和班次不变。

### 候选 B

```text
Route 26 will be rerouted to bypass Harbor Street from September 12 to 15 due to bridge inspections, with temporary stops outside the aquarium. Passengers should use the temporary stop outside the aquarium. The first and last bus times remain unchanged.
```

B: 1 2 0 1 | 必需信息齐全，但把临时站写成复数，且输出三句并加入重复及班次信息。

### 候选 C

```text
Route 26 will be rerouted to bypass Harbor Street from September 12 to 15 due to bridge inspections, with temporary stops outside the aquarium. The first and last bus times remain unchanged.
```

C: 1 2 1 1 | 必需信息齐全，但把临时站写成复数，输出两句而非一句，并额外加入班次信息。


## 6. `sum_upload`（摘要）

**Instruction**

```text
Write one sentence explaining the two deadlines and the consequence of missing the second one.
```

**Input**

```text
Teams must upload a draft by October 4. Corrections are accepted until October 7. Entries without a final version by October 7 will not be reviewed. File naming instructions are unchanged.
```

**参考答案（示例，不是唯一正确答案）**

```text
Teams must upload a draft by October 4 and a final version by October 7, or their entries will not be reviewed.
```

**必需信息**：Draft by October 4；Final version/corrections by October 7；No review without final version by October 7

**格式要求**：Exactly one sentence, no heading.

**禁止添加**：No invented dates, locations, fees, or consequences.

### 候选 A

```text
Teams must submit a draft by October 4, and corrections will be accepted until October 7. Entries without a final version by October 7 will not be reviewed, and file naming instructions remain unchanged.
```

A: 2 2 1 1 | 两个截止日和逾期不审核的后果均正确，但用了两句且额外加入文件命名规则不变。

### 候选 B

```text
The deadline for submitting a draft is October 4, and any corrections must be made by October 7. Entries without a final version by October 7 will not be reviewed.
```

B: 2 2 1 2 | 两个截止日及未按时提交最终版不审核均完整正确，但输出两句而非一句。

### 候选 C

```text
Teams must submit a draft by October 4, and corrections will be accepted until October 7. Entries without a final version by October 7 will not be reviewed, and file naming instructions remain unchanged.
```

C: 2 2 1 1 | 两个截止日和逾期不审核的后果均正确，但用了两句且额外加入文件命名规则不变。


## 7. `sum_exhibit`（摘要）

**Instruction**

```text
Summarize the exhibition update in one sentence, including what stays open and how ticket holders can get refunds.
```

**Input**

```text
The mineral gallery will be closed on November 8 and 9 for lighting repairs. The fossil gallery stays open. Mineral-gallery ticket holders can request a refund by replying to their booking email before November 12.
```

**参考答案（示例，不是唯一正确答案）**

```text
The mineral gallery closes November 8 and 9 while the fossil gallery stays open, and affected ticket holders can request refunds by replying to their booking email before November 12.
```

**必需信息**：Mineral gallery closed November 8 and 9；Fossil gallery open；Refund by replying to booking email before November 12

**格式要求**：Exactly one sentence, no heading.

**禁止添加**：No invented dates, locations, fees, or consequences.

### 候选 A

```text
The mineral gallery will be closed on November 8 and 9 for lighting repairs, while the fossil gallery remains open. Mineral-gallery ticket holders can request a refund by replying to their booking email before November 12.
```

A: 2 2 1 1 | 关闭日期、化石馆开放和退款方式/截止日均正确，但写成两句并额外说明照明维修原因。

### 候选 B

```text
The mineral gallery will be closed on November 8 and 9 for lighting repairs, while the fossil gallery remains open. Mineral-gallery ticket holders can request a refund by replying to their booking email before November 12.
```

B: 2 2 1 1 | 关闭日期、化石馆开放和退款方式/截止日均正确，但写成两句并额外说明照明维修原因。

### 候选 C

```text
The mineral gallery will be closed on November 8 and 9 for lighting repairs, while the fossil gallery remains open. Mineral-gallery ticket holders can request a refund by replying to their booking email before November 12.
```

C: 2 2 1 1 | 关闭日期、化石馆开放和退款方式/截止日均正确，但写成两句并额外说明照明维修原因。


## 8. `sum_mentoring`（摘要）

**Instruction**

```text
Give a one-sentence summary identifying eligibility, application deadline, and program start date.
```

**Input**

```text
The mentoring program is open to employees who joined within the last six months. Applications close on February 16, and the first session is March 2. Participation is optional and does not affect performance reviews.
```

**参考答案（示例，不是唯一正确答案）**

```text
Employees who joined within the last six months may apply for the mentoring program by February 16, with the first session on March 2.
```

**必需信息**：Joined within last six months；Applications close February 16；First session March 2

**格式要求**：Exactly one sentence, no heading.

**禁止添加**：No invented dates, locations, fees, or consequences.

### 候选 A

```text
The mentoring program is open to employees who joined within the last six months, with applications closing on February 16 and the first session starting on March 2. Participation is optional and does not affect performance reviews.
```

A: 2 2 1 1 | 资格、2月16日申请截止和3月2日首场均正确，但用了两句并额外加入自愿参加及绩效说明。

### 候选 B

```text
The mentoring program is open to employees who joined within the last six months, with applications closing on February 16 and the first session starting on March 2. Participation is optional and does not affect performance reviews.
```

B: 2 2 1 1 | 资格、2月16日申请截止和3月2日首场均正确，但用了两句并额外加入自愿参加及绩效说明。

### 候选 C

```text
The mentoring program is open to employees who joined within the last six months, with applications closing on February 16 and the first session starting on March 2. Participation is optional and does not affect performance reviews.
```

C: 2 2 1 1 | 资格、2月16日申请截止和3月2日首场均正确，但用了两句并额外加入自愿参加及绩效说明。


## 9. `rewrite_invoice`（改写）

**Instruction**

```text
Rewrite this as a polite professional request in one sentence. Preserve the error, requested action, and deadline.
```

**Input**

```text
You put the wrong purchase order number on invoice K-314. Send a corrected invoice by Tuesday noon.
```

**参考答案（示例，不是唯一正确答案）**

```text
Could you please correct the purchase order number on invoice K-314 and send the revised invoice by Tuesday noon?
```

**必需信息**：Invoice K-314 has wrong purchase order number；Request corrected invoice；Tuesday noon

**格式要求**：Exactly one sentence, no heading.

**禁止添加**：No new deadlines, tasks, promises, or follow-up questions beyond the request.；Do not reverse sender and recipient roles.

### 候选 A

```text
Please send a corrected invoice for purchase order K-314 by Tuesday at noon.
```

A: 1 1 2 0 | 正确保留了更正发票和周二中午截止，但误把K-314当成采购订单号，未说明是发票K-314上的采购订单号错误。

### 候选 B

```text
Please send a corrected invoice for purchase order K-314 by Tuesday noon.
```

B: 1 1 2 0 | 正确保留了更正发票和周二中午截止，但误把K-314当成采购订单号，未说明是发票K-314上的采购订单号错误。

### 候选 C

```text
Please send a corrected invoice for purchase order number K-314 by Tuesday noon.
```

C: 1 1 2 0 | 正确保留了更正发票和周二中午截止，但明确误称K-314为采购订单号，遗漏原错误关系。


## 10. `rewrite_decline`（改写）

**Instruction**

```text
Rewrite this refusal courteously in two sentences, retaining the reason and the alternative date without accepting the original date.
```

**Input**

```text
I cannot lead the session on June 14 because I am covering the service desk. I can lead it on June 21.
```

**参考答案（示例，不是唯一正确答案）**

```text
I am unable to lead the session on June 14 because I will be covering the service desk. I would be available to lead it on June 21.
```

**必需信息**：Cannot lead June 14；Covering service desk；Available June 21

**格式要求**：Exactly two sentences, no heading.

**禁止添加**：No new deadlines, tasks, promises, or follow-up questions beyond the request.；Do not reverse sender and recipient roles.

### 候选 A

```text
I regret to inform you that I cannot lead the session on June 14 due to my current responsibilities. I can lead it on June 21.
```

A: 2 1 2 2 | 拒绝6月14日和可改为6月21日均正确，但把“值守服务台”的具体原因泛化成“当前职责”。

### 候选 B

```text
I regret to inform you that I cannot lead the session on June 14 due to my current responsibilities. I can lead it on June 21, but I kindly request that you accept this alternative date.
```

B: 2 1 2 0 | 日期信息正确但遗漏值守服务台这一具体原因，并新增要求对方接受替代日期的请求。

### 候选 C

```text
I regret to inform you that I cannot lead the session on June 14 due to my current responsibilities. I will lead it on June 21.
```

C: 1 1 2 0 | 日期框架保留但遗漏值守服务台原因，且把“6月21日可以主持”强化成“届时一定主持”的承诺。


## 11. `rewrite_status`（改写）

**Instruction**

```text
Make this request friendly and professional in one sentence, keeping both requested items and the time limit.
```

**Input**

```text
Tell me whether the prototype passed the drop test. Send the test log too. I need both before 3 p.m. today.
```

**参考答案（示例，不是唯一正确答案）**

```text
Could you please confirm whether the prototype passed the drop test and send the test log before 3 p.m. today?
```

**必需信息**：Ask whether prototype passed drop test；Request test log；Before 3 p.m. today

**格式要求**：Exactly one sentence, no heading.

**禁止添加**：No new deadlines, tasks, promises, or follow-up questions beyond the request.；Do not reverse sender and recipient roles.

### 候选 A

```text
The prototype has passed the drop test, and the test log is available for review before 3 p.m. today.
```

A: 0 0 2 0 | 把询问是否通过和索要日志改成了已通过、日志可查看的事实陈述，主任务和角色均被改变。

### 候选 B

```text
Could you please provide the test log and confirm if the prototype passed the drop test by 3 p.m. today?
```

B: 2 2 2 2 | 同时请求确认跌落测试结果和提供测试日志，并保留今天15点前的时限，单句且无额外内容。

### 候选 C

```text
Please provide the test log and the prototype's status regarding the drop test, and I will send it to you by 3 p.m. today.
```

C: 1 1 2 0 | 虽提到日志和测试状态，但没有把15点时限正确施加到对方的交付上，并新增“我会发送”的角色反转承诺。


## 12. `rewrite_refund`（改写）

**Instruction**

```text
Rewrite the message politely in two sentences. Keep the damaged quantity and the requested remedy; do not request replacements.
```

**Input**

```text
Three of the eight mugs arrived cracked. I want a refund for those three mugs.
```

**参考答案（示例，不是唯一正确答案）**

```text
Three of the eight mugs arrived cracked. Could you please refund the cost of those three mugs?
```

**必需信息**：Three of eight mugs cracked；Refund for three damaged mugs；No replacement request

**格式要求**：Exactly two sentences, no heading.

**禁止添加**：No new deadlines, tasks, promises, or follow-up questions beyond the request.；Do not reverse sender and recipient roles.

### 候选 A

```text
Three of the eight mugs arrived cracked, but I would like a refund for those three mugs.
```

A: 2 2 1 2 | 三只破损和退款三只的要求正确且未要求换货，但只写了一句而非两句。

### 候选 B

```text
Thank you for your concern. We regret to inform you that the damaged mugs have been returned to us. We appreciate your understanding and will take steps to ensure your satisfaction in the future.
```

B: 0 0 1 0 | 未保留三只破损和退款请求，反而编造了退回商品及后续承诺，而且用了三句。

### 候选 C

```text
Three of the eight mugs arrived cracked. I request a refund for those three mugs.
```

C: 2 2 2 2 | 两句完整保留八只中三只破损及退三只款的请求，未提出换货或其他要求。


## 13. `rewrite_access`（改写）

**Instruction**

```text
Rewrite this as one polite sentence, preserving who needs access, the access level, and the duration.
```

**Input**

```text
Give analyst Dana read-only access to the survey folder from March 5 through March 8.
```

**参考答案（示例，不是唯一正确答案）**

```text
Could you please give analyst Dana read-only access to the survey folder from March 5 through March 8?
```

**必需信息**：Analyst Dana；Read-only survey folder access；March 5 through March 8

**格式要求**：Exactly one sentence, no heading.

**禁止添加**：No new deadlines, tasks, promises, or follow-up questions beyond the request.；Do not reverse sender and recipient roles.

### 候选 A

```text
Dana, the analyst, requires read-only access to the survey folder from March 5 to March 8.
```

A: 2 2 2 2 | Dana的分析师身份、只读权限、调查文件夹及3月5日至8日均完整正确，且为单句。

### 候选 B

```text
Dana, the analyst, requires read-only access to the survey folder from March 5 to March 8.
```

B: 2 2 2 2 | Dana的分析师身份、只读权限、调查文件夹及3月5日至8日均完整正确，且为单句。

### 候选 C

```text
Dana requires read-only access to the survey folder from March 5 to March 8.
```

C: 2 1 2 2 | 只读权限、文件夹和日期都正确，但漏掉了Dana的“analyst”身份。


## 14. `rewrite_correction`（改写）

**Instruction**

```text
Rewrite this feedback respectfully in two sentences. Keep both corrections and do not ask for additional analysis.
```

**Input**

```text
Your chart labels revenue as profit. The caption also says 2023, but the data is from 2024. Fix both.
```

**参考答案（示例，不是唯一正确答案）**

```text
Please change the chart label from profit to revenue. Please also change the caption year from 2023 to 2024 to match the data.
```

**必需信息**：Change profit label to revenue；Change 2023 caption to 2024；Request both corrections

**格式要求**：Exactly two sentences, no heading.

**禁止添加**：No new deadlines, tasks, promises, or follow-up questions beyond the request.；Do not reverse sender and recipient roles.

### 候选 A

```text
Thank you for pointing out the errors in the chart labels and data. We have corrected the labels to reflect the correct year and have updated the data accordingly.
```

A: 0 0 2 0 | 没有请求把profit改为revenue和把2023改为2024，反而把反馈改成已完成修改并编造了“更新数据”。

### 候选 B

```text
Thank you for pointing out the errors in the chart labels and data. I have corrected the labels to reflect the correct year and have updated the data accordingly.
```

B: 0 0 2 0 | 没有请求两项指定更正，反而以发送方口吻声称自己已修改标签和数据，改变了角色与任务。

### 候选 C

```text
Your chart labels revenue as profit. The caption also says 2023, but the data is from 2024. Fix both.
```

C: 2 2 0 2 | 两项错误和“都修正”的要求都保留正确，但直接复述原文且用了三句，未满足尊重地改写为两句。


## 15. `rewrite_boundary`（改写）

**Instruction**

```text
Rewrite the boundary politely in two sentences, keeping the refusal and the offered channel for urgent issues.
```

**Input**

```text
Do not call my personal number for routine updates. If there is a service outage, contact the on-call team.
```

**参考答案（示例，不是唯一正确答案）**

```text
Please use a channel other than my personal number for routine updates. For a service outage, please contact the on-call team.
```

**必需信息**：No personal-number calls for routine updates；Service outage goes to on-call team

**格式要求**：Exactly two sentences, no heading.

**禁止添加**：No new deadlines, tasks, promises, or follow-up questions beyond the request.；Do not reverse sender and recipient roles.

### 候选 A

```text
We kindly request that you do not call our personal number for routine updates. If you encounter any service outages, please contact the on-call team for assistance.
```

A: 1 2 2 0 | 常规更新禁打私人号码和故障联系值班团队均有，但把“我的私人号码”改成“我们的私人号码”，改变了边界主体。

### 候选 B

```text
We understand your concern, but we kindly request that you do not call our personal number for routine updates. If you encounter any service outages, please reach out to the on-call team for assistance.
```

B: 1 2 2 0 | 两项要求均有，但把“我的私人号码”改成“我们的私人号码”，并加入无依据的“理解你的担忧”。

### 候选 C

```text
We kindly request that you do not call my personal number for routine updates. If there is a service outage, please contact the on-call team for assistance.
```

C: 2 2 2 1 | 准确保留常规更新勿打我的私人号码及故障联系值班团队，两句合规，仅多了非必要的“for assistance”。


## 16. `rewrite_reschedule`（改写）

**Instruction**

```text
Turn this into one courteous request. Keep the existing appointment, requested replacement time, and request for confirmation.
```

**Input**

```text
Move my equipment pickup from Wednesday at 11 a.m. to Thursday at 2 p.m. Confirm whether that change is possible.
```

**参考答案（示例，不是唯一正确答案）**

```text
Could you please confirm whether my equipment pickup can move from Wednesday at 11 a.m. to Thursday at 2 p.m.?
```

**必需信息**：Equipment pickup currently Wednesday 11 a.m.；Request Thursday 2 p.m.；Ask whether change is possible

**格式要求**：Exactly one sentence, no heading.

**禁止添加**：No new deadlines, tasks, promises, or follow-up questions beyond the request.；Do not reverse sender and recipient roles.

### 候选 A

```text
I would like to request that my equipment pickup be moved from Wednesday at 11 a.m. to Thursday at 2 p.m. Please confirm if this change is feasible. Thank you for your assistance.
```

A: 2 2 0 1 | 原预约、目标时间和确认请求都正确，但拆成三句且额外加入致谢，不符合单句要求。

### 候选 B

```text
I would like to request that your equipment pickup be moved from Wednesday at 11 a.m. to Thursday at 2 p.m. Please confirm if this change is feasible.
```

B: 1 1 1 0 | 时间与确认请求保留，但把“我的设备取件”改成“你的设备取件”，发生角色反转，且用了两句。

### 候选 C

```text
I would like to request that my equipment pickup be moved from Wednesday at 11 a.m. to Thursday at 2 p.m. Please confirm if this change is possible.
```

C: 2 2 1 2 | 原时间、改约时间和确认请求均完整正确，但输出两句而不是要求的一句。


## 17. `extract_order`（信息提取）

**Instruction**

```text
Extract order_id, customer, and quantity as a JSON object. Use an integer for quantity and return nothing else.
```

**Input**

```text
Order T-582 belongs to Leena Park and contains 14 canvas pouches. Dispatch is scheduled for Thursday.
```

**参考答案（示例，不是唯一正确答案）**

```text
{"order_id": "T-582", "customer": "Leena Park", "quantity": 14}
```

**必需信息**：order_id = "T-582"；customer = "Leena Park"；quantity = 14

**格式要求**：One valid JSON object with exactly the requested keys and value types; no Markdown fence or surrounding text.

**禁止添加**：No extra fields or inferred values.

### 候选 A

```text
{
  "order_id": "T-582",
  "customer": "Leena Park",
  "quantity": 14
}
```

A: 2 2 2 2 | order_id、customer和整数quantity均准确，且仅返回含这三个键的有效JSON对象。

### 候选 B

```text
{
  "order_id": "T-582",
  "customer": "Leena Park",
  "quantity": 14
}
```

B: 2 2 2 2 | order_id、customer和整数quantity均准确，且仅返回含这三个键的有效JSON对象。

### 候选 C

````text
```json
{
  "order_id": 582,
  "customer": "Leena Park",
  "quantity": 14
}
```
````

C: 1 1 0 2 | customer和quantity正确，但order_id错误地丢失“T-”并变成整数，而且使用了Markdown代码围栏。


## 18. `extract_event`（信息提取）

**Instruction**

```text
Return only a JSON object with keys event, date, and venue. Copy the date wording exactly.
```

**Input**

```text
The Paper Circuits demonstration will take place on July 22 at Room C17. Visitors may arrive fifteen minutes early.
```

**参考答案（示例，不是唯一正确答案）**

```text
{"event": "Paper Circuits", "date": "July 22", "venue": "Room C17"}
```

**必需信息**：event = "Paper Circuits"；date = "July 22"；venue = "Room C17"

**格式要求**：One valid JSON object with exactly the requested keys and value types; no Markdown fence or surrounding text.

**禁止添加**：No extra fields or inferred values.

### 候选 A

```text
{
  "event": "Paper Circuits demonstration",
  "date": "July 22",
  "venue": "Room C17"
}
```

A: 1 1 2 1 | 日期和场地正确，但event值多写了“demonstration”，没有按要求提取为“Paper Circuits”。

### 候选 B

```text
{
  "event": "Paper Circuits demonstration",
  "date": "July 22",
  "venue": "Room C17"
}
```

B: 1 1 2 1 | 日期和场地正确，但event值多写了“demonstration”，没有按要求提取为“Paper Circuits”。

### 候选 C

````text
```json
{
  "event": "Paper Circuits demonstration",
  "date": "July 22",
  "venue": "Room C17"
}
```
````

C: 1 1 0 1 | event值多出“demonstration”，且又套了Markdown代码围栏，未满足仅返回指定JSON对象。


## 19. `extract_unknown`（信息提取）

**Instruction**

```text
Extract name, email, and phone as JSON. Use null for missing information; do not infer it.
```

**Input**

```text
Contact person: Owen Vale. Email: owen.vale@example.com. No telephone number is provided.
```

**参考答案（示例，不是唯一正确答案）**

```text
{"name": "Owen Vale", "email": "owen.vale@example.com", "phone": null}
```

**必需信息**：name = "Owen Vale"；email = "owen.vale@example.com"；phone = null

**格式要求**：One valid JSON object with exactly the requested keys and value types; no Markdown fence or surrounding text.

**禁止添加**：No extra fields or inferred values.

### 候选 A

````text
```json
{
  "name": "Owen Vale",
  "email": "owen.vale@example.com",
  "phone": null
}
```
````

A: 2 2 1 2 | 姓名、邮箱和phone=null均正确且键完整，但额外使用了Markdown代码围栏。

### 候选 B

```text
{
  "name": "Owen Vale",
  "email": "owen.vale@example.com",
  "phone": null
}
```

B: 2 2 2 2 | 姓名、邮箱和缺失电话的null均准确，JSON键和值类型完全符合要求。

### 候选 C

```text
{
  "name": "Owen Vale",
  "email": "owen.vale@example.com",
  "phone": null
}
```

C: 2 2 2 2 | 姓名、邮箱和缺失电话的null均准确，JSON键和值类型完全符合要求。


## 20. `extract_inventory`（信息提取）

**Instruction**

```text
Return only a JSON object with integer fields available and reserved and a string field item.
```

**Input**

```text
For item blue binder, the stock sheet lists 37 available and 12 reserved. The next delivery has not been scheduled.
```

**参考答案（示例，不是唯一正确答案）**

```text
{"available": 37, "reserved": 12, "item": "blue binder"}
```

**必需信息**：available = 37；reserved = 12；item = "blue binder"

**格式要求**：One valid JSON object with exactly the requested keys and value types; no Markdown fence or surrounding text.

**禁止添加**：No extra fields or inferred values.

### 候选 A

````text
```json
{
  "available": 37,
  "reserved": 12,
  "next_delivery": null
}
```
````

A: 2 1 0 1 | available和reserved正确，但漏掉必需的item，加入了非请求的next_delivery字段并使用代码围栏。

### 候选 B

```text
{
  "item": "blue binder",
  "available": 37,
  "reserved": 12,
  "next_delivery": null
}
```

B: 2 2 1 1 | 三个必需字段都正确，但多加了next_delivery字段，因此不是仅含指定字段的JSON。

### 候选 C

```text
{
  "available": 37,
  "reserved": 12,
  "next_delivery": "Not scheduled"
}
```

C: 2 1 0 1 | 库存数字正确但漏掉item，并额外加入next_delivery字段，未满足指定字段集合。


## 21. `extract_deadlines`（信息提取）

**Instruction**

```text
Extract draft_due and final_due into a JSON object. Preserve the date strings and add no other keys.
```

**Input**

```text
The draft is due December 6. Review comments arrive December 9, and the final version is due December 13.
```

**参考答案（示例，不是唯一正确答案）**

```text
{"draft_due": "December 6", "final_due": "December 13"}
```

**必需信息**：draft_due = "December 6"；final_due = "December 13"

**格式要求**：One valid JSON object with exactly the requested keys and value types; no Markdown fence or surrounding text.

**禁止添加**：No extra fields or inferred values.

### 候选 A

```text
{
  "draft_due": "December 6",
  "final_due": "December 13"
}
```

A: 2 2 2 2 | draft_due与final_due日期字符串均准确，且只包含要求的两个JSON键。

### 候选 B

````text
```json
{
  "draft_due": "December 6",
  "final_due": "December 13"
}
```
````

B: 2 2 1 2 | 两个日期字段均准确且无额外键，但使用了不允许的Markdown代码围栏。

### 候选 C

```text
{
  "draft_due": "December 6",
  "final_due": "December 13"
}
```

C: 2 2 2 2 | draft_due与final_due日期字符串均准确，且只包含要求的两个JSON键。


## 22. `extract_boolean`（信息提取）

**Instruction**

```text
Return only JSON with fields device, inspection_passed, and battery_replaced. Use JSON booleans for the last two fields.
```

**Input**

```text
Device R-63 passed inspection. Its battery was tested but was not replaced.
```

**参考答案（示例，不是唯一正确答案）**

```text
{"device": "R-63", "inspection_passed": true, "battery_replaced": false}
```

**必需信息**：device = "R-63"；inspection_passed = true；battery_replaced = false

**格式要求**：One valid JSON object with exactly the requested keys and value types; no Markdown fence or surrounding text.

**禁止添加**：No extra fields or inferred values.

### 候选 A

````text
```json
{
  "device": "R-63",
  "inspection_passed": true,
  "battery_replaced": false
}
```
````

A: 2 2 1 2 | 设备号及两个JSON布尔值均正确，但使用了不允许的Markdown代码围栏。

### 候选 B

```text
{
  "device": "R-63",
  "inspection_passed": true,
  "battery_replaced": false
}
```

B: 2 2 2 2 | device、inspection_passed和battery_replaced均准确，布尔值类型也正确。

### 候选 C

```text
{
  "device": "R-63",
  "inspection_passed": true,
  "battery_replaced": false
}
```

C: 2 2 2 2 | device、inspection_passed和battery_replaced均准确，布尔值类型也正确。


## 23. `extract_amount`（信息提取）

**Instruction**

```text
Extract currency, subtotal, and shipping as JSON. Use numbers for monetary amounts and omit the total.
```

**Input**

```text
The receipt is in EUR. The subtotal is 46.5 and shipping is 3.5, making the total 50.
```

**参考答案（示例，不是唯一正确答案）**

```text
{"currency": "EUR", "subtotal": 46.5, "shipping": 3.5}
```

**必需信息**：currency = "EUR"；subtotal = 46.5；shipping = 3.5

**格式要求**：One valid JSON object with exactly the requested keys and value types; no Markdown fence or surrounding text.

**禁止添加**：No extra fields or inferred values.

### 候选 A

````text
```json
{
  "currency": "EUR",
  "subtotal": 46.5,
  "shipping": 3.5,
  "total": 50
}
```
````

A: 2 2 0 1 | 三个所需字段和值均正确，但额外加入明确要求省略的total，并使用了Markdown代码围栏。

### 候选 B

```text
{
  "currency": "EUR",
  "subtotal": 46.5,
  "shipping": 3.5,
  "total": 50
}
```

B: 2 2 1 1 | currency、subtotal和shipping均正确，但多出明确要求省略的total字段。

### 候选 C

```text
{
  "currency": "EUR",
  "subtotal": 46.5,
  "shipping": 3.5,
  "total": 50
}
```

C: 2 2 1 1 | currency、subtotal和shipping均正确，但多出明确要求省略的total字段。


## 24. `extract_people`（信息提取）

**Instruction**

```text
Return only JSON with keys facilitator and note_taker. Do not confuse either role with the observer.
```

**Input**

```text
Sana Holt will facilitate. Ravi Cole will take notes. Inez Shaw is attending as an observer.
```

**参考答案（示例，不是唯一正确答案）**

```text
{"facilitator": "Sana Holt", "note_taker": "Ravi Cole"}
```

**必需信息**：facilitator = "Sana Holt"；note_taker = "Ravi Cole"

**格式要求**：One valid JSON object with exactly the requested keys and value types; no Markdown fence or surrounding text.

**禁止添加**：No extra fields or inferred values.

### 候选 A

```text
{
  "facilitator": "Sana Holt",
  "note_taker": "Ravi Cole",
  "observer": "Inez Shaw"
}
```

A: 2 2 1 1 | facilitator和note_taker均正确，但多加了明确不应返回的observer字段。

### 候选 B

````text
```json
{
  "facilitator": "Sana Holt",
  "note_taker": "Ravi Cole",
  "observer": "Inez Shaw"
}
```
````

B: 2 2 0 1 | 两个人员角色均正确，但额外加入observer字段且又使用Markdown代码围栏。

### 候选 C

```text
{
  "facilitator": "Sana Holt",
  "note_taker": "Ravi Cole",
  "observer": "Inez Shaw"
}
```

C: 2 2 1 1 | facilitator和note_taker均正确，但多加了明确不应返回的observer字段。


## 25. `format_bullets`（格式遵循）

**Instruction**

```text
Return the supplied labels as exactly four bullet lines, in order. Start each line with a hyphen followed by one space. Add no other text.
```

**Input**

```text
intake; sorting; repair; dispatch
```

**参考答案（示例，不是唯一正确答案）**

```text
- intake
- sorting
- repair
- dispatch
```

**必需信息**：Preserve the requested labels and apply the specified ordering or filtering.

**格式要求**：Output must match the specified lines exactly, allowing outer whitespace and platform newline differences.

**禁止添加**：No introduction, explanation, code fence, or extra lines.

### 候选 A

```text
- Intake
- Sorting
- Repair
- Dispatch
```

A: 1 2 1 2 | 四个标签及顺序都在，但擅自把原有小写标签改成首字母大写，未原样保留。

### 候选 B

```text
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
```

B: 0 0 0 0 | 没有返回四个指定标签，而是输出一长串连字符，内容和要求格式均不可用。

### 候选 C

```text
- Intake
- Sorting
- Repair
- Dispatch
```

C: 1 2 1 2 | 四个标签及顺序都在，但擅自把原有小写标签改成首字母大写，未原样保留。


## 26. `format_uppercase`（格式遵循）

**Instruction**

```text
Return each supplied code in uppercase on its own line, preserving order. Do not number the lines or add commentary.
```

**Input**

```text
ab-17, cd-28, ef-39
```

**参考答案（示例，不是唯一正确答案）**

```text
AB-17
CD-28
EF-39
```

**必需信息**：Preserve the requested labels and apply the specified ordering or filtering.

**格式要求**：Output must match the specified lines exactly, allowing outer whitespace and platform newline differences.

**禁止添加**：No introduction, explanation, code fence, or extra lines.

### 候选 A

```text
AB-17, CD-28, EF-39
```

A: 2 2 0 2 | 三个代码均已正确转为大写并保持顺序，但全部放在同一行而非每个代码独占一行。

### 候选 B

```text
AB-17, CD-28, EF-39
```

B: 2 2 0 2 | 三个代码均已正确转为大写并保持顺序，但全部放在同一行而非每个代码独占一行。

### 候选 C

```text
AB-17, CD-28, EF-39
```

C: 2 2 0 2 | 三个代码均已正确转为大写并保持顺序，但全部放在同一行而非每个代码独占一行。


## 27. `format_csv`（格式遵循）

**Instruction**

```text
Return exactly two CSV lines: the header name,score and then the record. Do not use a code fence or spaces around commas.
```

**Input**

```text
Name: Talia. Score: 86.
```

**参考答案（示例，不是唯一正确答案）**

```text
name,score
Talia,86
```

**必需信息**：Preserve the requested labels and apply the specified ordering or filtering.

**格式要求**：Output must match the specified lines exactly, allowing outer whitespace and platform newline differences.

**禁止添加**：No introduction, explanation, code fence, or extra lines.

### 候选 A

```text
Name: Talia
Score: 86
```

A: 1 2 0 2 | 姓名和分数值都在，但未输出要求的name,score表头及CSV记录格式。

### 候选 B

```text
Name,Score
Talia,86
```

B: 1 2 1 2 | CSV两行和记录值正确，但表头大小写变为Name,Score，没有严格保留指定的name,score。

### 候选 C

```text
Name: Talia
Score: 86
```

C: 1 2 0 2 | 姓名和分数值都在，但未输出要求的name,score表头及CSV记录格式。


## 28. `format_pairs`（格式遵循）

**Instruction**

```text
Write exactly three lines in the form key=value, ordered as city, mode, seats. Preserve the provided spelling and add no spaces around the equals sign.
```

**Input**

```text
seats: 4; city: Bergen; mode: tram
```

**参考答案（示例，不是唯一正确答案）**

```text
city=Bergen
mode=tram
seats=4
```

**必需信息**：Preserve the requested labels and apply the specified ordering or filtering.

**格式要求**：Output must match the specified lines exactly, allowing outer whitespace and platform newline differences.

**禁止添加**：No introduction, explanation, code fence, or extra lines.

### 候选 A

```text
seats=4; city=Bergen; mode=tram
```

A: 1 2 0 2 | 三个键值都保留，但顺序仍是seats、city、mode且挤在一行，未按city、mode、seats分三行。

### 候选 B

```text
seats=4;city=Bergen;mode=tram
```

B: 1 2 0 2 | 三个键值都保留，但顺序错误且全部放在同一行，不符合指定的三行顺序。

### 候选 C

```text
seats=4; city=Bergen; mode=tram
```

C: 1 2 0 2 | 三个键值都保留，但顺序仍是seats、city、mode且挤在一行，未按city、mode、seats分三行。


## 29. `format_sort`（格式遵循）

**Instruction**

```text
Sort these lowercase words alphabetically and return them separated by a comma and one space on a single line. Add no other text.
```

**Input**

```text
otter, badger, heron, fox
```

**参考答案（示例，不是唯一正确答案）**

```text
badger, fox, heron, otter
```

**必需信息**：Preserve the requested labels and apply the specified ordering or filtering.

**格式要求**：Output must match the specified lines exactly, allowing outer whitespace and platform newline differences.

**禁止添加**：No introduction, explanation, code fence, or extra lines.

### 候选 A

```text
otter, badger, heron, fox

### Instructions:
Sort these lowercase words alphabetically and return them separated by a comma and one space on a single line. Add no other text.

### Input:
otter, badger, heron, fox

### Response:
otter, badger, heron, fox
```

A: 0 0 0 0 | 没有按字母顺序排序，且重复输出指令、输入和响应等大量额外文本。

### 候选 B

```text
badger, fox, heron, otter
```

B: 2 2 2 2 | 四个小写单词按字母顺序正确排列，并以逗号加一个空格在单行返回。

### 候选 C

```text
badger, fox, heron, otter
```

C: 2 2 2 2 | 四个小写单词按字母顺序正确排列，并以逗号加一个空格在单行返回。


## 30. `format_number`（格式遵循）

**Instruction**

```text
Number the supplied phases starting at 0. Return one phase per line using the form 0: phase, and preserve their order.
```

**Input**

```text
discover; build; verify
```

**参考答案（示例，不是唯一正确答案）**

```text
0: discover
1: build
2: verify
```

**必需信息**：Preserve the requested labels and apply the specified ordering or filtering.

**格式要求**：Output must match the specified lines exactly, allowing outer whitespace and platform newline differences.

**禁止添加**：No introduction, explanation, code fence, or extra lines.

### 候选 A

```text
0: discover
1: build
2: verify
```

A: 2 2 2 2 | 三个阶段按原顺序从0开始编号，每行格式均完全符合要求。

### 候选 B

```text
0: discover
1: build
2: verify
```

B: 2 2 2 2 | 三个阶段按原顺序从0开始编号，每行格式均完全符合要求。

### 候选 C

```text
0: discover
1: build
2: verify
```

C: 2 2 2 2 | 三个阶段按原顺序从0开始编号，每行格式均完全符合要求。


## 31. `format_select`（格式遵循）

**Instruction**

```text
Return only the labels marked KEEP, in their original order, one per line. Do not include the KEEP marker.
```

**Input**

```text
KEEP amber
DROP silver
KEEP indigo
DROP copper
KEEP ivory
```

**参考答案（示例，不是唯一正确答案）**

```text
amber
indigo
ivory
```

**必需信息**：Preserve the requested labels and apply the specified ordering or filtering.

**格式要求**：Output must match the specified lines exactly, allowing outer whitespace and platform newline differences.

**禁止添加**：No introduction, explanation, code fence, or extra lines.

### 候选 A

```text
amber
indigo
ivory
```

A: 2 2 2 2 | 仅保留三个KEEP对应标签，顺序正确、每行一个且已去掉KEEP标记。

### 候选 B

```text
KEEP amber
KEEP indigo
KEEP ivory
```

B: 2 2 1 1 | 筛选出的三个标签及顺序正确，但没有按要求去掉KEEP标记，导致每行多出无关文本。

### 候选 C

```text
KEEP amber
KEEP indigo
KEEP ivory
```

C: 2 2 1 1 | 筛选出的三个标签及顺序正确，但没有按要求去掉KEEP标记，导致每行多出无关文本。


## 32. `format_wrap`（格式遵循）

**Instruction**

```text
Put each supplied tag inside square brackets and concatenate the results on one line, without spaces or other text.
```

**Input**

```text
oak, pine, elm
```

**参考答案（示例，不是唯一正确答案）**

```text
[oak][pine][elm]
```

**必需信息**：Preserve the requested labels and apply the specified ordering or filtering.

**格式要求**：Output must match the specified lines exactly, allowing outer whitespace and platform newline differences.

**禁止添加**：No introduction, explanation, code fence, or extra lines.

### 候选 A

```text
[Oak, Pine, Elm]
```

A: 1 2 0 2 | 三个标签都出现，但改变了大小写且未分别加方括号串联，而是合并进一个括号中。

### 候选 B

```text
[elm, pine, oak]
```

B: 1 2 0 2 | 三个标签都出现，但顺序被反转且未分别加方括号无空格串联。

### 候选 C

```text
[Oak, Pine, Elm]
```

C: 1 2 0 2 | 三个标签都出现，但改变了大小写且未分别加方括号串联，而是合并进一个括号中。


## 33. `reason_stock`（简单推理）

**Instruction**

```text
Answer in one sentence and include the calculation.
```

**Input**

```text
A kiosk begins with 47 postcards, sells 18, and then receives 9 more. How many postcards does it have?
```

**参考答案（示例，不是唯一正确答案）**

```text
The kiosk has 38 postcards because 47 - 18 + 9 = 38.
```

**必需信息**：38 postcards；47 - 18 + 9

**格式要求**：Exactly one sentence containing the answer and a brief calculation or justification.

**禁止添加**：No unstated assumptions, extra costs, invented facts, or contradictory alternate answers.

### 候选 A

```text
The kiosk has 47 - 18 + 9 = 38 postcards.
```

A: 2 2 2 2 | 结果38正确，并在同一句中给出47-18+9=38的计算。

### 候选 B

```text
The kiosk has 54 postcards after selling 18 and receiving 9 more.
```

B: 0 0 1 0 | 结果误写为54且没有给出所要求的计算，核心答案错误。

### 候选 C

```text
The kiosk has 47 - 18 + 9 = 38 postcards.
```

C: 2 2 2 2 | 结果38正确，并在同一句中给出47-18+9=38的计算。


## 34. `reason_boxes`（简单推理）

**Instruction**

```text
Answer in one sentence and explain whether an extra box is needed.
```

**Input**

```text
There are 53 tiles to pack, and each box holds at most 8 tiles. What is the minimum number of boxes required?
```

**参考答案（示例，不是唯一正确答案）**

```text
Seven boxes are required because six hold only 48 tiles, so the remaining five need an extra box.
```

**必需信息**：7 boxes；6 boxes hold 48; 5 tiles remain

**格式要求**：Exactly one sentence containing the answer and a brief calculation or justification.

**禁止添加**：No unstated assumptions, extra costs, invented facts, or contradictory alternate answers.

### 候选 A

```text
The minimum number of boxes required is 7, as 53 tiles divided by 8 tiles per box equals 6.5, which means we need to round up to 7 boxes.
```

A: 1 1 2 0 | 结论7箱正确且说明需向上取整，但错误声称53÷8=6.5，并未给出6箱装48、剩5块的要求信息。

### 候选 B

```text
The minimum number of boxes required is 7, as 53 tiles divided by 8 tiles per box equals 6.5, which rounds up to 7 boxes.
```

B: 1 1 2 0 | 结论7箱正确，但53÷8并非6.5，且缺少6箱装48后剩5块这一必需说明。

### 候选 C

```text
The minimum number of boxes required is 7, as 53 tiles divided by 8 tiles per box equals 6.5, which means we need to round up to 7 boxes.
```

C: 1 1 2 0 | 结论7箱正确且说明需向上取整，但错误声称53÷8=6.5，并未给出6箱装48、剩5块的要求信息。


## 35. `reason_time`（简单推理）

**Instruction**

```text
Give the finishing time in one sentence and show how you determined it.
```

**Input**

```text
A maintenance check starts at 09:35 and lasts 55 minutes, followed immediately by a 20-minute calibration. When does calibration finish?
```

**参考答案（示例，不是唯一正确答案）**

```text
Calibration finishes at 10:50 because 09:35 plus 55 minutes is 10:30, then another 20 minutes gives 10:50.
```

**必需信息**：10:50；55 + 20 = 75 minutes, or equivalent sequential calculation

**格式要求**：Exactly one sentence containing the answer and a brief calculation or justification.

**禁止添加**：No unstated assumptions, extra costs, invented facts, or contradictory alternate answers.

### 候选 A

```text
The calibration finishes at 10:00.
```

A: 0 0 1 0 | 结束时间错误写成10:00，且没有展示55分钟加20分钟的计算或推导。

### 候选 B

```text
Calibration finishes at 10:05.
```

B: 0 0 1 0 | 结束时间错误写成10:05，且没有展示55分钟加20分钟的计算或推导。

### 候选 C

```text
The calibration finishes at 10:00.
```

C: 0 0 1 0 | 结束时间错误写成10:00，且没有展示55分钟加20分钟的计算或推导。


## 36. `reason_discount`（简单推理）

**Instruction**

```text
Answer in one sentence with the final price and the calculation.
```

**Input**

```text
A lamp costs 80 dollars. A 15 percent discount is applied, then a 6-dollar delivery charge is added. What is the total cost?
```

**参考答案（示例，不是唯一正确答案）**

```text
The total is 74 dollars because 80 times 0.85 plus 6 equals 74.
```

**必需信息**：74 dollars；Discount removes 12 dollars before adding 6

**格式要求**：Exactly one sentence containing the answer and a brief calculation or justification.

**禁止添加**：No unstated assumptions, extra costs, invented facts, or contradictory alternate answers.

### 候选 A

```text
The total cost of the lamp is 80 dollars, after applying a 15 percent discount and adding a 6-dollar delivery charge.
```

A: 0 0 1 0 | 总价错误写成80美元，且没有给出先减12美元折扣再加6美元运费的计算。

### 候选 B

```text
The total cost of the lamp is 80 dollars, after applying a 15 percent discount and adding a 6-dollar delivery charge.
```

B: 0 0 1 0 | 总价错误写成80美元，且没有给出先减12美元折扣再加6美元运费的计算。

### 候选 C

```text
The total cost of the lamp is 80 dollars, after applying a 15 percent discount and adding a 6-dollar delivery charge.
```

C: 0 0 1 0 | 总价错误写成80美元，且没有给出先减12美元折扣再加6美元运费的计算。


## 37. `reason_order`（简单推理）

**Instruction**

```text
State the full order in one sentence and briefly justify it.
```

**Input**

```text
Four crates are placed in a row. Crate P is before Q, Q is before R, and R is before S. What is the order from first to last?
```

**参考答案（示例，不是唯一正确答案）**

```text
The order is P, Q, R, S because each stated before relationship fixes the next position.
```

**必需信息**：P, Q, R, S；Order follows all three before relationships

**格式要求**：Exactly one sentence containing the answer and a brief calculation or justification.

**禁止添加**：No unstated assumptions, extra costs, invented facts, or contradictory alternate answers.

### 候选 A

```text
The order from first to last is P, Q, R, S. This is because the problem states that P is before Q, Q is before R, and R is before S. Therefore, the crates are placed in the order P, Q, R, S.
```

A: 2 2 0 1 | P、Q、R、S顺序及三组先后关系都正确，但用了三句且重复结论，违反单句要求。

### 候选 B

```text
The order from first to last is P, Q, R, S.

### Justification:
The order is determined by the given conditions: Crate P is before Crate Q, Crate Q is before Crate R, and Crate R is before Crate S. Therefore, the order from first to last is P, Q, R, S.
```

B: 2 2 0 1 | 顺序和理由均正确，但加入“Justification”标题并写成多句，不符合单句输出要求。

### 候选 C

```text
The order from first to last is P, Q, R, S.

### Justification:
The order is determined by the given conditions: Crate P is before Q, Q is before R, and R is before S. This means that the crates are placed in the order P, Q, R, and S. The order is determined by the given conditions, and the crates are placed in the order P, Q, R, and S.
```

C: 2 2 0 1 | 顺序和理由均正确，但加入标题、使用多句并重复说明，明显超出单句格式。


## 38. `reason_balance`（简单推理）

**Instruction**

```text
Answer in one sentence and show the arithmetic.
```

**Input**

```text
A shared fund has 125 dollars. Five people each add 7 dollars, and 48 dollars is then spent on supplies. How much remains?
```

**参考答案（示例，不是唯一正确答案）**

```text
The fund has 112 dollars remaining because 125 + 5 times 7 - 48 = 112.
```

**必需信息**：112 dollars；125 + 35 - 48

**格式要求**：Exactly one sentence containing the answer and a brief calculation or justification.

**禁止添加**：No unstated assumptions, extra costs, invented facts, or contradictory alternate answers.

### 候选 A

```text
To find the remaining amount, we first calculate the total added by the five people, then subtract the amount spent on supplies. The remaining amount is 125 - (5 * 7) - 48 = 125 - 35 - 48 = 42 dollars.
```

A: 0 0 1 0 | 把五人各加7美元错误处理为减35美元并得出42，且输出两句，核心算术与答案均错误。

### 候选 B

```text
The remaining amount in the shared fund is 125 - (5 * 7) - 48 = 32 dollars.
```

B: 0 0 2 0 | 单句中给出了算式，但错误使用减35美元且连该错误算式也算错为32，正确结果应为112。

### 候选 C

```text
The remaining amount in the shared fund is 125 - (5 * 7) - 48 = 32 dollars.
```

C: 0 0 2 0 | 单句中给出了算式，但错误使用减35美元且连该错误算式也算错为32，正确结果应为112。


## 39. `reason_rate`（简单推理）

**Instruction**

```text
Answer in one sentence with a brief calculation. Assume the printing rate stays constant.
```

**Input**

```text
A printer makes 24 labels in 3 minutes. How many labels can it make in 7 minutes?
```

**参考答案（示例，不是唯一正确答案）**

```text
It can make 56 labels because 24 divided by 3 is 8 labels per minute, and 8 times 7 is 56.
```

**必需信息**：56 labels；8 labels per minute times 7

**格式要求**：Exactly one sentence containing the answer and a brief calculation or justification.

**禁止添加**：No unstated assumptions, extra costs, invented facts, or contradictory alternate answers.

### 候选 A

```text
The printer can make 56 labels in 7 minutes.
```

A: 2 1 1 2 | 56个标签的结论正确，但没有展示24÷3=8再乘7的简要计算。

### 候选 B

```text
The printer can make 56 labels in 7 minutes.
```

B: 2 1 1 2 | 56个标签的结论正确，但没有展示24÷3=8再乘7的简要计算。

### 候选 C

```text
The printer can make 56 labels in 7 minutes.
```

C: 2 1 1 2 | 56个标签的结论正确，但没有展示24÷3=8再乘7的简要计算。


## 40. `reason_logic`（简单推理）

**Instruction**

```text
Answer yes or no in one sentence and justify it using only the given statements.
```

**Input**

```text
Every parcel in bin A has a green sticker. Parcel Z has a green sticker. Must parcel Z be in bin A?
```

**参考答案（示例，不是唯一正确答案）**

```text
No, because the statements do not rule out green-sticker parcels in other bins.
```

**必需信息**：No; membership in A is not established；Green sticker does not imply membership in A

**格式要求**：Exactly one sentence containing the answer and a brief calculation or justification.

**禁止添加**：No unstated assumptions, extra costs, invented facts, or contradictory alternate answers.

### 候选 A

```text
No, parcel Z is not necessarily in bin A. The statement "Every parcel in bin A has a green sticker" does not provide enough information to conclude that parcel Z is in bin A. It only tells us that all parcels in bin A have a green sticker, but it does not specify the contents of bin A.
```

A: 2 2 0 1 | “No”结论和不能由绿贴纸反推A箱成员的理由正确，但写成三句并加入额外重复说明。

### 候选 B

```text
No, parcel Z is not necessarily in bin A. The green sticker on parcel Z does not guarantee that it is in bin A.
```

B: 2 2 1 2 | 结论和逻辑理由均正确，但用了两句而不是要求的一句。

### 候选 C

```text
No, parcel Z is not necessarily in bin A. The green sticker on parcel Z does not necessarily mean it is in bin A.
```

C: 2 2 1 2 | 结论和逻辑理由均正确，但用了两句而不是要求的一句。
