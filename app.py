
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="ML Explorer",
    page_icon="📊",
    layout="wide"
)

st.title("ML Explorer")
st.write(
    "An interactive explorer for heart disease classification, "
    "clustering and wholesale customer segmentation."
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Home",
        "Heart Disease: EDA",
        "Heart Disease: Classification",
        "Wholesale Customers: EDA",
        "Wholesale Customer Segmentation"
    ]
)


if page == "Home":
    st.subheader("Welcome")
    st.write(
        "Explore the Cleveland Heart Disease dataset through "
        "exploratory data analysis and classification, then "
        "discover customer segments using the Wholesale Customers dataset."
    )

    st.markdown("### 23CSE301 - Case Study Project")
    st.write("**H Sahana** [CB.SC.U4CSE24415]")
    st.write("**Sheethal Raj** [CB.SC.U4CSE24446]")
    st.write("**Swethambari V** [CB.SC.U4CSE24452]")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Datasets", "2")

    with col2:
        st.metric("Project Modules", "3")



elif page == "Heart Disease: EDA":
    st.header("Heart Disease: Exploratory Data Analysis")

    columns = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal", "num"
    ]

    try:
        df = pd.read_csv(
            "processed.cleveland.data",
            names=columns,
            na_values="?"
        )

        st.subheader("Dataset Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Patient records", len(df))
        col2.metric("Features", len(df.columns) - 1)
        col3.metric("Missing values", int(df.isnull().sum().sum()))

        st.subheader("Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe().round(2), use_container_width=True)

        st.subheader("Feature Distribution")
        feature = st.selectbox(
            "Choose a feature",
            df.columns.tolist()
        )

        st.bar_chart(df[feature].value_counts().sort_index())

        st.subheader("Heart Disease Target Distribution")
        target_counts = df["num"].value_counts().sort_index()
        st.bar_chart(target_counts)

        st.subheader("Correlation Heatmap")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax
        )
        st.pyplot(fig)
        plt.close(fig)

    except FileNotFoundError:
        st.error(
            "Could not find processed.cleveland.data. "
            "Make sure it is in the same folder as app.py."
        )



elif page == "Heart Disease: Classification":
    st.header("Heart Disease Classification")
    st.write(
        "Explore the Cleveland dataset and compare five "
        "classification algorithms from our notebook."
    )

    # Dataset
    @st.cache_data
    def load_heart_data():
        columns = [
            "age", "sex", "cp", "trestbps", "chol",
            "fbs", "restecg", "thalach", "exang",
            "oldpeak", "slope", "ca", "thal", "num"
        ]
        return pd.read_csv(
            "processed.cleveland.data",
            names=columns,
            na_values="?"
        )

    df = load_heart_data()

    # Dataset overview
    c1, c2, c3 = st.columns(3)
    c1.metric("Patient records", len(df))
    c2.metric("Input features", df.shape[1] - 1)
    c3.metric("Missing values", int(df.isna().sum().sum()))

    st.divider()

    # Notebook's actual comparison results
    st.subheader("Classification Model Comparison")

    results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "KNN",
            "Gaussian Naive Bayes",
            "Decision Tree",
            "SVC"
        ],
        "Accuracy": [0.574, 0.607, 0.639, 0.574, 0.557],
        "Precision": [0.527, 0.551, 0.592, 0.566, 0.430],
        "Recall": [0.574, 0.607, 0.639, 0.574, 0.557],
        "Weighted F1": [0.548, 0.572, 0.613, 0.566, 0.484],
        "ROC-AUC": [0.866, 0.799, 0.879, 0.719, 0.846]
    })

    st.dataframe(
        results.style.format({
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "Weighted F1": "{:.3f}",
            "ROC-AUC": "{:.3f}"
        }),
        hide_index=True,
        width="stretch"
    )

    best_model = results.loc[results["Accuracy"].idxmax()]

    st.success(
        f"Best accuracy among these five models: "
        f"{best_model['Model']} ({best_model['Accuracy']:.3f})"
    )

    st.divider()

    # Interactive metric chart
    st.subheader("Visual Model Comparison")

    metric = st.selectbox(
        "Choose a metric",
        ["Accuracy", "Precision", "Recall", "Weighted F1", "ROC-AUC"]
    )

    chart_data = results.set_index("Model")[[metric]].sort_values(
        by=metric,
        ascending=False
    )

    st.bar_chart(chart_data)

    st.divider()

    # Dataset details
    with st.expander("Explore the dataset"):
        st.dataframe(df.head(10), width="stretch")

        st.subheader("Target distribution")
        st.bar_chart(df["num"].value_counts().sort_index())

        st.subheader("Feature statistics")
        st.dataframe(df.describe().T, width="stretch")

    st.caption(
        "Results are reproduced from classification_B1.ipynb. "
        "They are reference results, not predictions from a model "
        "trained by this Streamlit page."
    )


elif page == "Wholesale Customers: EDA":
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.header("Wholesale Customers: Exploratory Data Analysis")
    st.write(
        "Explore customer spending patterns, identify outliers, "
        "and examine relationships between product categories."
    )

    try:
        df = pd.read_csv("Wholesale customers data.csv")
    except FileNotFoundError:
        st.error(
        "Could not find Wholesale customers data.csv. "
        "Make sure it is in the same folder as app.py."
    )
        st.stop()


    spending_cols = [
        "Fresh", "Milk", "Grocery", "Frozen",
        "Detergents_Paper", "Delicassen"
    ]

    # Dataset overview
    st.subheader("1. Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df))
    col2.metric("Product Categories", len(spending_cols))
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.write("First five records")
    st.dataframe(df.head())

    with st.expander("View descriptive statistics"):
        st.dataframe(df[spending_cols].describe().round(2))

    # Spending distributions
    st.subheader("2. Spending Distribution")

    selected_category = st.selectbox(
        "Choose a product category",
        spending_cols
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(
        data=df,
        x=selected_category,
        bins=30,
        kde=True,
        ax=ax
    )
    ax.set_title(f"Distribution of {selected_category} Spending")
    ax.set_xlabel("Annual spending")
    ax.set_ylabel("Number of customers")
    st.pyplot(fig)
    plt.close(fig)

    # Outlier analysis
    st.subheader("3. Outlier Analysis")

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df[spending_cols], orient="h", ax=ax)
    ax.set_title("Spending Outliers Across Product Categories")
    ax.set_xlabel("Annual spending")
    st.pyplot(fig)
    plt.close(fig)

    # Correlation analysis
    st.subheader("4. Correlation Between Spending Categories")

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        df[spending_cols].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )
    ax.set_title("Spending Correlation Heatmap")
    st.pyplot(fig)
    plt.close(fig)

    # Channel and Region
    st.subheader("5. Customer Distribution")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Customers by Channel")
        st.bar_chart(df["Channel"].value_counts().sort_index())

    with col2:
        st.write("Customers by Region")
        st.bar_chart(df["Region"].value_counts().sort_index())


elif page == "Wholesale Customer Segmentation":
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score,
        calinski_harabasz_score,
        adjusted_rand_score
    )

    st.header("Wholesale Customer Segmentation")
    st.write(
        "Compare two clustering algorithms and explore how wholesale "
        "customers group according to their annual spending patterns."
    )

    # Load dataset
    try:
        df = pd.read_csv("Wholesale customers data.csv")
    except FileNotFoundError:
        st.error("Could not find Wholesale customers data.csv.")
        st.stop()

    spend_cols = [
        "Fresh", "Milk", "Grocery", "Frozen",
        "Detergents_Paper", "Delicassen"
    ]

    # Dataset overview
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Customer records", len(df))
    col2.metric("Input features", len(spend_cols))
    col3.metric("Clusters", 2)

    st.divider()

    # Preprocessing: log transformation and scaling
    X_log = np.log1p(df[spend_cols])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    # Fit clustering algorithms
    kmeans = KMeans(
        n_clusters=2,
        init="k-means++",
        n_init=10,
        random_state=42
    )
    kmeans_labels = kmeans.fit_predict(X_scaled)

    ward = AgglomerativeClustering(
        n_clusters=2,
        metric="euclidean",
        linkage="ward"
    )
    ward_labels = ward.fit_predict(X_scaled)

    # Comparison table
    st.subheader("Clustering Algorithm Comparison")

    results = pd.DataFrame({
        "Algorithm": ["K-Means", "Agglomerative (Ward)"],
        "Silhouette Score": [
            silhouette_score(X_scaled, kmeans_labels),
            silhouette_score(X_scaled, ward_labels)
        ],
        "Davies-Bouldin Index": [
            davies_bouldin_score(X_scaled, kmeans_labels),
            davies_bouldin_score(X_scaled, ward_labels)
        ],
        "Calinski-Harabasz Index": [
            calinski_harabasz_score(X_scaled, kmeans_labels),
            calinski_harabasz_score(X_scaled, ward_labels)
        ]
    })

    st.dataframe(
        results.style.format({
            "Silhouette Score": "{:.3f}",
            "Davies-Bouldin Index": "{:.3f}",
            "Calinski-Harabasz Index": "{:.3f}"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Higher Silhouette and Calinski-Harabasz scores are preferable. "
        "Lower Davies-Bouldin scores are preferable."
    )

    # Highlight the preferred algorithm
    st.subheader("Preferred Clustering Algorithm")

    st.success(
        "K-Means performs better on all three comparison metrics "
        "in this run. Ward linkage remains a useful hierarchical "
        "clustering alternative."
    )

    st.divider()

    # Interactive metric comparison
    st.subheader("Visual Algorithm Comparison")

    metric = st.selectbox(
        "Choose an evaluation metric",
        [
            "Silhouette Score",
            "Davies-Bouldin Index",
            "Calinski-Harabasz Index"
        ]
    )

    chart_data = results.set_index("Algorithm")[[metric]]

    st.bar_chart(chart_data)

    st.divider()

    # Select algorithm for detailed exploration
    st.subheader("Explore Cluster Results")

    algorithm = st.selectbox(
        "Choose an algorithm to explore",
        ["K-Means", "Agglomerative (Ward)"],
        key="segmentation_algorithm"
    )

    labels = (
        kmeans_labels
        if algorithm == "K-Means"
        else ward_labels
    )

    with st.expander("Cluster sizes and spending profiles", expanded=True):
        sizes = pd.Series(labels).value_counts().sort_index()
        sizes.index = [f"Cluster {i}" for i in sizes.index]
        st.write("Number of customers in each cluster")
        st.bar_chart(sizes)

        profiles = df[spend_cols].copy()
        profiles["Cluster"] = labels
        profile_means = profiles.groupby("Cluster")[spend_cols].mean()
        profile_means.index = [
            f"Cluster {i}" for i in profile_means.index
        ]

        st.write("Average annual spending by cluster")
        st.dataframe(
            profile_means.round(2),
            use_container_width=True
        )

    with st.expander("PCA cluster visualization"):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Cluster": [f"Cluster {i}" for i in labels]
        })

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.scatterplot(
            data=pca_df,
            x="PC1",
            y="PC2",
            hue="Cluster",
            palette="Set1",
            s=55,
            alpha=0.8,
            ax=ax
        )
        ax.set_title(f"{algorithm}: PCA Projection")
        st.pyplot(fig)
        plt.close(fig)

        st.write(
            f"The first two principal components explain "
            f"{pca.explained_variance_ratio_.sum() * 100:.1f}% "
            "of the variance."
        )

    with st.expander("Post-hoc validation"):
        st.write(
            "Channel and Region were excluded from clustering. "
            "The following scores compare the discovered clusters "
            "with those existing labels after fitting."
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Adjusted Rand Index vs Channel",
            f"{adjusted_rand_score(df['Channel'], labels):.3f}"
        )

        col2.metric(
            "Adjusted Rand Index vs Region",
            f"{adjusted_rand_score(df['Region'], labels):.3f}"
        )

    with st.expander("Explore the dataset"):
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Descriptive statistics")
        st.dataframe(
            df[spend_cols].describe().T.round(2),
            use_container_width=True
        )
