--- LLMからフラグ抽出＋WHERE用条件カラム追加版（フラグV2）
CREATE TABLE Demo.DischargeSummaryDoc (
  DocId               BIGINT PRIMARY KEY,

  -- 匿名化した患者ID（"Patient/1" でも UUID でもOK）
  PatientId           VARCHAR(64) NOT NULL,

  AdmitDate           DATE,
  DischargeDate        DATE,

  -- 絞り込み（A）
  AgeBand             VARCHAR(16),      -- "0-17","18-39","40-64","65+"
  Sex                 VARCHAR(8),       -- "male","female","other","unknown"
  Department          VARCHAR(64),      -- "Cardiology" 等
  DischargeDisposition VARCHAR(32),     -- "home","transfer","facility","death" 等

  -- コード系：最初はCSVでOK（後で正規化テーブルに分離してもよい）
  DiagnosisCodes      VARCHAR(2000),    -- 例: "ICD10:I50.9,ICD10:E11.9"
  ProcedureCodes      VARCHAR(2000),    -- 例: "K-code:K546,..." / SNOMED / ICD-10-PCS 等

  -- 原本保持（Cのため）
  SourceFilename      VARCHAR(255),     -- 添付ファイル名
  SourceUri           VARCHAR(1000),    -- ファイル保存先URI（任意）
  SourceText          VARCHAR(32000),   -- 全文をDBに置くなら（大きいなら外部に逃がしてURIだけでもOK）

  -- 運用用
  Lang                VARCHAR(10) NOT NULL DEFAULT 'ja',
  UpdateTimeStamp  TIMESTAMP DEFAULT CURRENT_TIMESTAMP(9) ON UPDATE CURRENT_TIMESTAMP(9),

  -- 任意：文書のハッシュ（重複取り込み防止）
  SourceHash          VARCHAR(128)
)
go

-- 絞り込みで使う索引（必要最低限）
CREATE INDEX PatientIdIdx ON Demo.DischargeSummaryDoc(PatientId)
go

CREATE INDEX DischargeDateIdx ON Demo.DischargeSummaryDoc(DischargeDate)
go

CREATE INDEX DepartmentIdx ON Demo.DischargeSummaryDoc(Department)
go
CREATE INDEX AgeBandIdx ON Demo.DischargeSummaryDoc(AgeBand)
go

CREATE INDEX DispositionIdx ON Demo.DischargeSummaryDoc(DischargeDisposition)
go

CREATE TABLE Demo.DischargeSummaryChunk (
  DocKey          VARCHAR(10),
  DocId           BIGINT NOT NULL,

  -- セクション種別（最初は4つ固定が楽）
  -- "discharge_dx" 退院時診断 / "hospital_course"入院経過 / "plan_followup"退院時方針・フォロー / "discharge_meds"退院時薬剤
  SectionType     VARCHAR(50) NOT NULL,

  SectionTitle    VARCHAR(200),          -- 見出しが取れるなら
  SectionText     VARCHAR(32000) NOT NULL,

  -- 原文参照（C：根拠提示やハイライトに使う）
  StartOffset     INTEGER,               -- SourceText内の開始位置（取れなければNULLでOK）
  EndOffset       INTEGER,

  -- SectionText（今は入院経過）のベクトル
  Embedding       VECTOR(Float,1536) NOT NULL,
  
  -- LLMから抽出したFlag情報
  FlagsJson VARCHAR(10000),

  --**** ケアレベル・臓器サポート系フラグ(LLMから抽出したフラグを格納)
  -- 集中治療室（ICU）での管理を行ったか
  -- 一般病棟ではなく、重症患者向けの集中治療室に入室して治療したことを示す
  HasICUCare  BIT,
  -- 酸素投与を行ったか
  -- 鼻カニュラやマスクなどで酸素を補助的に投与した（人工呼吸器は含まない）
  HasOxygenTherapy  BIT,
  -- HFNC（高流量鼻カニュラ）を使用したか
  -- 通常の酸素投与より高流量・高濃度の酸素を鼻から供給する呼吸補助
  HasHFNC BIT,
  -- NPPV（非侵襲的陽圧換気）を使用したか
  -- マスクを用いた呼吸補助（CPAP/BiPAPなど）、挿管はしていない
  HasNPPV BIT,
  -- 気管挿管を行ったか
  -- 口や鼻から気管にチューブを入れて気道を確保した処置
  HasIntubation BIT,
  -- 人工呼吸器管理を行ったか
  -- 挿管後に人工呼吸器（機械換気）で呼吸を管理した
  HasMechanicalVentilation  BIT,
  -- 透析治療を行ったか
  -- 腎機能低下に対して血液透析や持続的血液浄化（CHDF/CRRTなど）を実施した
  HasDialysis BIT,
  -- 昇圧剤を使用したか
  -- 血圧低下に対してノルアドレナリン等の薬剤で循環を維持した
  HasVasopressor  BIT,
  -- ***　ここまで

  UpdateTimeStamp  TIMESTAMP DEFAULT CURRENT_TIMESTAMP(9) ON UPDATE CURRENT_TIMESTAMP(9),

  CONSTRAINT DocIdFK FOREIGN KEY (DocId) REFERENCES Demo.DischargeSummaryDoc(DocId)
)
go

CREATE INDEX DocIdIdx ON Demo.DischargeSummaryChunk(DocId)
go

CREATE INDEX SectionTypeIdx ON Demo.DischargeSummaryChunk(SectionType)
go

CREATE INDEX HasICUCareIdx ON Demo.DischargeSummaryChunk(HasICUCare)
go

CREATE INDEX HasOxygenTherapyIdx ON Demo.DischargeSummaryChunk(HasOxygenTherapy)
go

CREATE INDEX HasHFNCIdx ON Demo.DischargeSummaryChunk(HasHFNC)
go

CREATE INDEX HasNPPVIdx ON Demo.DischargeSummaryChunk(HasNPPV)
go

CREATE INDEX HasIntubationIdx ON Demo.DischargeSummaryChunk(HasIntubation)
go

CREATE INDEX HasMechanicalVentilation ON Demo.DischargeSummaryChunk(HasMechanicalVentilation)
go

CREATE INDEX HasDialysisIdx ON Demo.DischargeSummaryChunk(HasDialysis)
go

CREATE INDEX HasVasopressorIdx ON Demo.DischargeSummaryChunk(HasVasopressor)
go

CREATE INDEX EmbeddingHNSWIndex ON TABLE Demo.DischargeSummaryChunk (Embedding)
     AS HNSW(Distance='DotProduct')
go
