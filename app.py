import streamlit as st
import pandas as pd
import joblib

# 页面设置
st.set_page_config(
    page_title="感染性心内膜炎脑梗预测系统",
    page_icon="🧠",
    layout="centered"
)

# 标题
st.title("🧠 感染性心内膜炎脑梗风险预测系统")

st.markdown("---")

st.write("请输入患者临床指标信息")

# 加载模型
model = joblib.load("adaboost_model.pkl")

# 输入区域
年龄 = st.number_input("年龄（岁）", 0, 100, 50)

LDH = st.number_input("LDH（U/L）", 0.0, 10000.0, 300.0)

D_二聚体 = st.number_input("D-二聚体（mg/L）", 0.0, 20.0, 1.0)

白蛋白 = st.number_input("白蛋白（g/L）", 0.0, 100.0, 35.0)

AST_ALT = st.number_input("AST/ALT", 0.0, 20.0, 1.0)

金葡菌 = st.selectbox(
    "是否金黄色葡萄球菌感染",
    ["否", "是"]
)

栓塞史 = st.selectbox(
    "是否存在栓塞史",
    ["否", "是"]
)

# 编码
金葡菌值 = 1 if 金葡菌 == "是" else 0
栓塞值 = 1 if 栓塞史 == "是" else 0

# 预测按钮
if st.button("开始预测"):

    X = pd.DataFrame(
        [[
            年龄,
            LDH,
            D_二聚体,
            白蛋白,
            AST_ALT,
            金葡菌值,
            栓塞值
        ]],
        columns=[
            "Age",
            "LDH",
            "D_dimer",
            "Albumin",
            "AST_ALT",
            "Staph",
            "Embolism"
        ]
    )

    # 预测概率
    prob = model.predict_proba(X)[0][1]

    st.markdown("---")

    st.subheader("预测结果")

    st.metric(
        label="脑梗发生风险",
        value=f"{prob:.1%}"
    )

    # 风险分层
    if prob < 0.2:
        st.success("低风险")
    elif prob < 0.5:
        st.warning("中风险")
    else:
        st.error("高风险")

    st.info("本系统仅供临床辅助决策参考")