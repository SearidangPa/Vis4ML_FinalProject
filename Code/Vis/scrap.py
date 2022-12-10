# Histogram of distance between each pair of attribution methods  
distance_signed_rank = pd.DataFrame(columns=['shap_lime', 'shap_ig', 'shap_deepLift', 'lime_ig', 'lime_deepLift', 'ig_deepLift'])
for i in range (len(method_names)):
    for j in range (len(method_names)):
        if i < j:
            distance_signed_rank[method_names[i] + '_' + method_names[j]] = ((rank_dfs[i] - rank_dfs[j]) ** 2).sum(axis=1) ** 0.5
tsne_filename = '../Saved/Proj/tsne.pkl'
tsne_embedded_df = pd.read_pickle(tsne_filename)
combined = pd.concat([tsne_embedded_df, distance_signed_rank], axis = 1)
combined
brush = alt.selection(type="interval")

base = alt.Chart(combined).encode(
    opacity = alt.condition(brush, alt.value(1.0), alt.value(0.1))
).add_selection(
    brush
).properties(
    width=350,
    height=350
)

view_shap = base.mark_circle().encode(x='shap_x-tsne', y='shap_y-tsne', color = 'target:N',).properties(title="Click and drag to create a selection region")
view_lime = base.mark_circle().encode(x='lime_x-tsne', y='lime_y-tsne', color = 'target:N',).properties(title="Click and drag to create a selection region")
view_ig = base.mark_circle().encode(x='ig_x-tsne', y='ig_y-tsne', color = 'target:N',).properties(title="Click and drag to create a selection region")
view_deepLift = base.mark_circle().encode(x='deepLift_x-tsne', y='deepLift_y-tsne', color = 'target:N',).properties(title="Click and drag to create a selection region")
view_emb_data = base.mark_circle().encode(x='data_x-tsne', y='data_y-tsne', color = 'target:N',).properties(title="Click and drag to create a selection region")

view1 = alt.hconcat(view_shap, view_lime)
view2 = alt.hconcat(view_ig, view_deepLift)
t_sne_attr_plot = alt.vconcat(view1, view2)
view_histogram = alt.Chart(combined).mark_bar().encode(
    x= alt.X('shap_ig', bin=alt.Bin(step = 1, extent=[0, 30])), 
    y='count()'
).transform_filter(brush)

# view_mean_distance
t_sne_attr_plot | view_histogram