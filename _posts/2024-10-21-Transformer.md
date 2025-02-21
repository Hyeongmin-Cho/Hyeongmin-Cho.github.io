---
title: "[논문구현] Transformer (트랜스포머) 스크래치 구현부터 한영 번역 학습까지(Attention is All You Need)"
author: Hyeongmin Cho
date: 2024-10-21
categories: [Deep Learning, Transformer]
tags: [tech]
description: "Transformer 논문 'Attention is All You Need'를 스크래치로 구현하고 한영 번역 모델을 학습시키는 과정을 다룹니다."
robots: "index, follow"
pin: true
math: true

redirect_from:
  - /jekyll/2024-10-21-Transformer.html
---

# Transformer from Scratch in Pytorch

Large Language Model (LLM)이 최근 유행하고 있는데, 그 기반이 되는 Transformer 관련 내용을 정리하고자 본 포스트를 작성하게 되었습니다.

모델 구현, 학습 데이터 선정 및 구성, 토크나이저 학습, 모델 학습까지 차근차근 다뤄볼 생각입니다.

데이터 셋 구성 및 학습과 관련된 코드는 아래 깃허브 레포를 참고해주세요.
- [GitHub Repository](https://github.com/Hyeongmin-Cho/Transformer-from-Scratch-in-Pytorch)



## 목차
1. Transformer (트랜스포머) 구조 설명
2. 구현
   1. Embedding 및 Positional Encoding
   2. Padding Mask & Look-Ahead Mask 구현
   3. Multi-head Attention 구현
   4. Encoder 구현
   5. Decoder 구현
   6. Transformer 구현
3. 한영 번역 데이터 셋 선정
4. 토크나이저 학습
5. 모델 학습

## 1. Transformer (트랜스포머) 구조 설명

작성중...
