---
marp: true
theme: phishing_minimal
paginate: true
size: 16:9
title: LIME Phishing Detection
description: Minimal Marp deck distilled from blog_post.md
---

<style>
/* @theme phishing_minimal */
@import 'default';

:root {
  --paper: #f8f4ea;
  --ink: #111820;
  --muted: #5f665f;
  --line: #d6cdbf;
  --phish: #d9442e;
  --safe: #1b8f5a;
  --amber: #b8860b;
  --panel: #fffdf8;
}

section {
  background: var(--paper);
  color: var(--ink);
  font-family: "Avenir Next", "Gill Sans", "Trebuchet MS", sans-serif;
  font-size: 30px;
  letter-spacing: 0;
  padding: 66px 76px;
}

section::after {
  color: var(--muted);
  font-size: 18px;
}

h1, h2 {
  font-family: "DIN Condensed", "Avenir Next Condensed", "Impact", sans-serif;
  letter-spacing: 0;
  margin: 0;
  text-transform: uppercase;
}

h1 {
  font-size: 92px;
  line-height: .9;
  max-width: 900px;
}

h2 {
  border-bottom: 3px solid var(--phish);
  font-size: 60px;
  line-height: .95;
  margin-bottom: 34px;
  padding-bottom: 14px;
}

p, li {
  line-height: 1.22;
}

strong {
  color: var(--phish);
}

code {
  background: #fff0c8;
  border-radius: 4px;
  color: var(--ink);
  font-family: "Menlo", "SFMono-Regular", monospace;
  font-size: .78em;
  padding: .08em .26em;
}

pre {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 6px;
  font-size: 22px;
}

img {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 6px;
}

.subtitle {
  color: var(--muted);
  font-size: 32px;
  margin-top: 28px;
  max-width: 820px;
}

.authors {
  bottom: 62px;
  color: var(--muted);
  font-size: 22px;
  position: absolute;
}

.tag {
  color: var(--phish);
  font-family: "Menlo", "SFMono-Regular", monospace;
  font-size: 20px;
  font-weight: 700;
  letter-spacing: .08em;
  margin-bottom: 20px;
  text-transform: uppercase;
}

.grid-2 {
  display: grid;
  gap: 32px;
  grid-template-columns: 1fr 1fr;
}

.grid-3 {
  display: grid;
  gap: 22px;
  grid-template-columns: repeat(3, 1fr);
}

.stat {
  background: var(--panel);
  border: 1px solid var(--line);
  border-top: 7px solid var(--phish);
  border-radius: 6px;
  padding: 22px 24px;
}

.stat.safe {
  border-top-color: var(--safe);
}

.stat.amber {
  border-top-color: var(--amber);
}

.num {
  display: block;
  font-family: "DIN Condensed", "Avenir Next Condensed", "Impact", sans-serif;
  font-size: 68px;
  line-height: .9;
}

.label {
  color: var(--muted);
  display: block;
  font-size: 19px;
  margin-top: 8px;
  text-transform: uppercase;
}

.flow {
  display: grid;
  gap: 16px;
  grid-template-columns: repeat(5, 1fr);
  margin-top: 34px;
}

.step {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 6px;
  min-height: 126px;
  padding: 18px;
}

.step b {
  color: var(--phish);
  display: block;
  font-size: 21px;
  margin-bottom: 12px;
  text-transform: uppercase;
}

.step span {
  color: var(--muted);
  font-size: 22px;
}

.email {
  background: var(--panel);
  border-left: 8px solid var(--phish);
  border-radius: 6px;
  font-family: Georgia, serif;
  font-size: 36px;
  line-height: 1.18;
  padding: 28px 32px;
}

.tokens {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 24px 28px;
}

.tokens p {
  margin: 0 0 16px;
}

.tokens p:last-child {
  margin-bottom: 0;
}

.phish {
  color: var(--phish);
  font-weight: 800;
}

.safe {
  color: var(--safe);
  /* font-weight: 800; */
}

.note {
  color: var(--muted);
  font-size: 24px;
}

.wide img {
  max-height: 500px;
  object-fit: contain;
  width: 100%;
}

.curve img {
  max-height: 455px;
  object-fit: contain;
  width: 100%;
}

.image-pair {
  align-items: center;
  display: grid;
  gap: 24px;
  grid-template-columns: 1fr 1fr;
}

.image-pair img {
  max-height: 470px;
  object-fit: contain;
  width: 100%;
}
</style>

<!-- _paginate: false -->

<div class="tag">Explainable AI</div>

# Phishing Email Detection with LIME

<div class="subtitle">A DistilRoBERTa classifier predicts phishing. LIME explains the local evidence behind one email at a time.</div>

<div class="authors">Kirill Greshnov · Vladimir Paskal</div>

---

## Study Overview

<div class="grid-2">
<div>

**Problem:** binary email classification  
`1` phishing · `0` safe

**Need:** security analysts need reasons, not only a model score.

</div>
<div>

**Approach:** train a transformer, then explain single predictions with a model-agnostic local surrogate.

</div>
</div>

<div class="flow">
  <div class="step"><b>Data</b><span>clean labels</span></div>
  <div class="step"><b>Split</b><span>stratified</span></div>
  <div class="step"><b>Model</b><span>DistilRoBERTa</span></div>
  <div class="step"><b>LIME</b><span>token weights</span></div>
  <div class="step"><b>Check</b><span>deletion test</span></div>
</div>

---

## Data

<div class="grid-3">
<div class="stat amber">
  <span class="num">18,631</span>
  <span class="label">usable emails</span>
</div>
<div class="stat safe">
  <span class="num">11,322</span>
  <span class="label">safe emails</span>
</div>
<div class="stat">
  <span class="num">7,309</span>
  <span class="label">phishing emails</span>
</div>
</div>

<div class="grid-3" style="margin-top: 28px;">
<div class="stat">
  <span class="num">14,904</span>
  <span class="label">train rows</span>
</div>
<div class="stat">
  <span class="num">1,863</span>
  <span class="label">validation rows</span>
</div>
<div class="stat">
  <span class="num">1,864</span>
  <span class="label">test rows</span>
</div>
</div>

<p class="note">The source file has 18,650 rows; 19 blank-text rows are dropped before training. Class ratio is preserved across splits.</p>

---

## Model

<div class="grid-2">
<div>

Fine-tuned `distilroberta-base` for one epoch using Hugging Face Transformers.

<p class="note">Strong test metrics are useful, but they do not explain a specific alert.</p>

</div>
<div class="grid-2">
<div class="stat">
  <span class="num">0.9667</span>
  <span class="label">accuracy</span>
</div>
<div class="stat">
  <span class="num">0.9579</span>
  <span class="label">F1-score</span>
</div>
<div class="stat safe">
  <span class="num">0.9502</span>
  <span class="label">precision</span>
</div>
<div class="stat safe">
  <span class="num">0.9658</span>
  <span class="label">recall</span>
</div>
</div>
</div>

---

## LIME

LIME explains one prediction by learning a simple local model around the original email.

<div class="flow">
  <div class="step"><b>Tokenize</b><span>words + punctuation</span></div>
  <div class="step"><b>Mask</b><span>remove tokens randomly</span></div>
  <div class="step"><b>Score</b><span>ask transformer</span></div>
  <div class="step"><b>Weight</b><span>near samples higher</span></div>
  <div class="step"><b>Fit</b><span>ridge regression</span></div>
</div>

<div class="tokens" style="margin-top: 34px;">
<p><span class="phish">Positive weight</span> means the token increases phishing probability.</p>
<p><span class="safe">Negative weight</span> means the token decreases phishing probability in this local explanation.</p>
</div>

---

## Example

<div class="email">
Your account has been suspended. Click here to verify.
</div>

<div class="grid-2" style="margin-top: 30px;">
<div class="stat">
  <span class="num">0.9655</span>
  <span class="label">predicted phishing probability</span>
</div>
<div class="tokens">
<p><span class="phish">Phishing tokens:</span> 
Your · Click · here · verify</p>
<p><span class="safe">Safe-side local tokens:</span>
 has · account · been</p>
</div>
</div>

<p class="note">A token’s sign is local. It is not a global dictionary of safe or malicious words.</p>

---

## LIME output

<div class="image-pair">

![LIME highlighted email](./lime_html.jpg)

![LIME console output](./lime_results.jpg)

</div>

---

## Token importance

<div class="wide">

![Token importance chart](./artifacts/explanations/figures/token_importance.png)

</div>

---

## Deletion test

<div class="grid-2">
<div>

Remove high-importance tokens first and compare the phishing score against random deletion.

<div class="grid-2" style="margin-top: 24px;">
<div class="stat">
  <span class="num">0.4595</span>
  <span class="label">ranked AUC</span>
</div>
<div class="stat safe">
  <span class="num">0.8118</span>
  <span class="label">random AUC</span>
</div>
</div>

<p class="note">Lower AUC is better. The ranked curve drops faster over 30 phishing emails.</p>

</div>
<div class="curve">

![Deletion curve](./artifacts/deletion_curve.png)

</div>
</div>

---

## Limitations

<div class="grid-2">
<div>

- Word removal can create unnatural emails.
- Correlated phishing phrases can split credit.
- LIME explains one prediction, not the whole model.

</div>
</div>

<p class="note">Useful next step: run the deletion test directly with LIME-ranked tokens.</p>
