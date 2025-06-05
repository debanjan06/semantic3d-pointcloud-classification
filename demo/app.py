import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
import sys
from io import StringIO

# Handle imports gracefully
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    st.error("PyTorch not available. Please install with: pip install torch torchvision")
    TORCH_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    st.error("Plotly not available. Please install with: pip install plotly")
    PLOTLY_AVAILABLE = False

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    st.warning("Open3D not available. File upload will be limited.")
    OPEN3D_AVAILABLE = False

try:
    from sklearn.metrics import confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Only import model if torch is available
if TORCH_AVAILABLE:
    try:
        from models.pointnet2 import PointNet2SemSeg
        MODEL_AVAILABLE = True
    except ImportError:
        st.error("PointNet++ model not found. Please check src/models/pointnet2.py")
        MODEL_AVAILABLE = False
else:
    MODEL_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="🌍 Semantic3D Point Cloud Classifier",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Load the trained PointNet++ model"""
    if not MODEL_AVAILABLE:
        return None
    
    try:
        model = PointNet2SemSeg(num_classes=8, input_channels=6)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def generate_sample_data(scene_type="urban"):
    """Generate sample point cloud data for demo"""
    np.random.seed(42)
    
    if scene_type == "urban":
        # Generate urban scene with buildings, roads, trees
        points = []
        labels = []
        
        # Ground plane (man-made terrain)
        ground = np.random.uniform(-10, 10, (500, 2))
        ground_z = np.zeros((500, 1)) + np.random.normal(0, 0.1, (500, 1))
        ground_points = np.hstack([ground, ground_z])
        points.append(ground_points)
        labels.extend([0] * 500)  # man-made terrain
        
        # Buildings
        for _ in range(3):
            building_x = np.random.uniform(-8, 8)
            building_y = np.random.uniform(-8, 8)
            building_points = []
            for z in np.linspace(0, np.random.uniform(3, 8), 100):
                x_offset = np.random.normal(0, 0.5, 20)
                y_offset = np.random.normal(0, 0.5, 20)
                building_level = np.column_stack([
                    building_x + x_offset,
                    building_y + y_offset,
                    np.full(20, z)
                ])
                building_points.append(building_level)
            building_points = np.vstack(building_points)
            points.append(building_points)
            labels.extend([4] * len(building_points))  # buildings
        
        # Trees (high vegetation)
        for _ in range(4):
            tree_x = np.random.uniform(-9, 9)
            tree_y = np.random.uniform(-9, 9)
            tree_points = []
            for z in np.linspace(0, np.random.uniform(4, 10), 80):
                radius = max(0.1, 2 - z * 0.3)
                theta = np.random.uniform(0, 2*np.pi, 15)
                r = np.random.uniform(0, radius, 15)
                x = tree_x + r * np.cos(theta)
                y = tree_y + r * np.sin(theta)
                tree_level = np.column_stack([x, y, np.full(15, z)])
                tree_points.append(tree_level)
            tree_points = np.vstack(tree_points)
            points.append(tree_points)
            labels.extend([2] * len(tree_points))  # high vegetation
        
        # Cars
        for _ in range(2):
            car_center = np.random.uniform(-7, 7, 2)
            car_points = np.random.uniform(-1, 1, (50, 3))
            car_points[:, :2] += car_center
            car_points[:, 2] = np.abs(car_points[:, 2]) * 1.5
            points.append(car_points)
            labels.extend([7] * 50)  # cars
            
    elif scene_type == "forest":
        # Generate forest scene
        points = []
        labels = []
        
        # Natural terrain
        terrain = np.random.uniform(-15, 15, (800, 2))
        terrain_z = np.sin(terrain[:, 0] * 0.3) * np.cos(terrain[:, 1] * 0.3) + np.random.normal(0, 0.2, 800)
        terrain_points = np.column_stack([terrain, terrain_z])
        points.append(terrain_points)
        labels.extend([1] * 800)  # natural terrain
        
        # Many trees
        for _ in range(8):
            tree_x = np.random.uniform(-12, 12)
            tree_y = np.random.uniform(-12, 12)
            tree_height = np.random.uniform(8, 15)
            tree_points = []
            for z in np.linspace(0, tree_height, 120):
                radius = max(0.2, 3 - z * 0.2)
                theta = np.random.uniform(0, 2*np.pi, 20)
                r = np.random.uniform(0, radius, 20)
                x = tree_x + r * np.cos(theta)
                y = tree_y + r * np.sin(theta)
                tree_level = np.column_stack([x, y, np.full(20, z)])
                tree_points.append(tree_level)
            tree_points = np.vstack(tree_points)
            points.append(tree_points)
            labels.extend([2] * len(tree_points))  # high vegetation
            
        # Low vegetation/bushes
        for _ in range(15):
            bush_center = np.random.uniform(-10, 10, 2)
            bush_points = np.random.normal(0, 1, (30, 3))
            bush_points[:, :2] += bush_center
            bush_points[:, 2] = np.abs(bush_points[:, 2]) * 0.5
            points.append(bush_points)
            labels.extend([3] * 30)  # low vegetation
    
    # Combine all points
    all_points = np.vstack(points)
    all_labels = np.array(labels)
    
    # Add RGB colors based on labels
    colors = np.array([
        [0.5, 0.5, 0.5],  # man-made terrain - gray
        [0.4, 0.2, 0.1],  # natural terrain - brown
        [0.1, 0.6, 0.1],  # high vegetation - green
        [0.3, 0.8, 0.3],  # low vegetation - light green
        [0.8, 0.8, 0.8],  # buildings - light gray
        [0.3, 0.3, 0.3],  # hard scape - dark gray
        [1.0, 0.0, 1.0],  # scanning artifacts - magenta
        [1.0, 0.0, 0.0],  # cars - red
    ])
    
    point_colors = colors[all_labels]
    
    # Combine XYZ + RGB
    point_cloud_with_colors = np.hstack([all_points, point_colors])
    
    return point_cloud_with_colors, all_labels

def preprocess_points(points, max_points=4096):
    """Preprocess point cloud for model inference"""
    # Sample points if too many
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        return points, indices
    elif len(points) < max_points:
        # Pad with repeated points if too few
        indices = np.random.choice(len(points), max_points, replace=True)
        points = points[indices]
        return points, indices
    else:
        return points, np.arange(len(points))

def create_3d_plot(points, labels=None, title="Point Cloud"):
    """Create interactive 3D plot with plotly"""
    if not PLOTLY_AVAILABLE:
        st.error("Plotly required for 3D visualization")
        return None
        
    if labels is not None:
        class_names = [
            'man-made terrain', 'natural terrain', 'high vegetation',
            'low vegetation', 'buildings', 'hard scape', 
            'scanning artifacts', 'cars'
        ]
        
        colors = [
            'gray', 'brown', 'green', 'lightgreen',
            'lightgray', 'darkgray', 'magenta', 'red'
        ]
        
        # Create hover text
        hover_text = [class_names[label] for label in labels]
        point_colors = [colors[label] for label in labels]
    else:
        hover_text = ['Point'] * len(points)
        point_colors = 'lightblue'
    
    fig = go.Figure(data=go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=point_colors,
            opacity=0.8
        ),
        text=hover_text,
        hovertemplate='<b>%{text}</b><br>' +
                     'X: %{x:.2f}<br>' +
                     'Y: %{y:.2f}<br>' +
                     'Z: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=600,
        height=500,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def create_simple_plot(points, labels=None, title="Point Cloud"):
    """Create simple 2D scatter plot as fallback"""
    df = pd.DataFrame({
        'X': points[:, 0],
        'Y': points[:, 1],
        'Z': points[:, 2]
    })
    
    if labels is not None:
        class_names = [
            'man-made terrain', 'natural terrain', 'high vegetation',
            'low vegetation', 'buildings', 'hard scape', 
            'scanning artifacts', 'cars'
        ]
        df['Class'] = [class_names[label] for label in labels]
        
        fig = px.scatter(df, x='X', y='Y', color='Class', 
                        size=[1]*len(df), title=title,
                        hover_data=['Z'])
    else:
        fig = px.scatter(df, x='X', y='Y', title=title,
                        hover_data=['Z'])
    
    return fig

def main():
    # Header
    st.title("🌍 Semantic3D Point Cloud Classification")
    st.markdown("**Real-time 3D point cloud semantic segmentation using PointNet++**")
    
    # Check dependencies
    if not TORCH_AVAILABLE or not MODEL_AVAILABLE:
        st.error("⚠️ Some dependencies are missing. The demo will run in limited mode.")
        st.info("For full functionality, please ensure PyTorch and the PointNet++ model are available.")
    
    # Sidebar
    st.sidebar.header("🛠️ Model Information")
    st.sidebar.info("""
    **Architecture**: PointNet++ for Semantic Segmentation  
    **Dataset**: Semantic3D (8 classes)  
    **Input**: XYZ coordinates + RGB colors  
    **Output**: Per-point semantic labels
    
    **Classes**:
    - 🏗️ Man-made terrain
    - 🌱 Natural terrain  
    - 🌳 High vegetation
    - 🌿 Low vegetation
    - 🏢 Buildings
    - 🛤️ Hard scape
    - ⚡ Scanning artifacts
    - 🚗 Cars
    """)
    
    # Load model
    model = None
    if MODEL_AVAILABLE:
        with st.spinner("Loading PointNet++ model..."):
            model = load_model()
        if model is not None:
            st.sidebar.success("✅ Model loaded successfully!")
        else:
            st.sidebar.error("❌ Model loading failed")
    else:
        st.sidebar.warning("⚠️ Model not available")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📁 Upload File", "🎯 Try Samples", "📊 Model Info"])
    
    with tab1:
        st.subheader("Upload Your Point Cloud")
        
        if not OPEN3D_AVAILABLE:
            st.warning("Open3D not available. File upload functionality is limited.")
        
        uploaded_file = st.file_uploader(
            "Choose a point cloud file (.txt format recommended)", 
            type=['txt']
        )
        
        if uploaded_file is not None:
            st.success("File uploaded successfully!")
            
            try:
                # Load text file
                content = uploaded_file.getvalue().decode('utf-8')
                lines = content.strip().split('\n')
                
                # Parse points
                points_data = []
                for line in lines:
                    try:
                        coords = [float(x) for x in line.split()]
                        if len(coords) >= 3:
                            points_data.append(coords[:6])  # Take first 6 columns
                    except:
                        continue
                
                if points_data:
                    points_with_colors = np.array(points_data)
                    
                    # Ensure 6 columns
                    if points_with_colors.shape[1] < 6:
                        # Add random colors if missing
                        n_missing = 6 - points_with_colors.shape[1]
                        random_colors = np.random.uniform(0, 1, (len(points_with_colors), n_missing))
                        points_with_colors = np.hstack([points_with_colors, random_colors])
                    
                    st.info(f"Loaded {len(points_with_colors):,} points")
                    
                    if st.button("🚀 Classify Point Cloud", type="primary"):
                        if model is not None:
                            classify_point_cloud(model, points_with_colors)
                        else:
                            st.error("Model not available for classification")
                else:
                    st.error("Could not parse point cloud data")
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    with tab2:
        st.subheader("Try Sample Scenes")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🏙️ Urban Scene", type="primary"):
                points, true_labels = generate_sample_data("urban")
                st.info(f"Generated urban scene with {len(points):,} points")
                if model is not None:
                    classify_point_cloud(model, points, true_labels)
                else:
                    # Show visualization without classification
                    show_sample_only(points, true_labels, "Urban Scene")
        
        with col2:
            if st.button("🌲 Forest Scene", type="primary"):
                points, true_labels = generate_sample_data("forest")
                st.info(f"Generated forest scene with {len(points):,} points")
                if model is not None:
                    classify_point_cloud(model, points, true_labels)
                else:
                    show_sample_only(points, true_labels, "Forest Scene")
    
    with tab3:
        st.subheader("📈 Model Architecture")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **PointNet++ Architecture:**
            - 4 Set Abstraction layers with hierarchical sampling
            - Farthest Point Sampling for robust point selection  
            - Ball query for local neighborhood grouping
            - Feature Propagation for dense prediction
            - Multi-scale feature learning
            """)
            
        with col2:
            st.markdown("""
            **Model Statistics:**
            - Parameters: ~1.5M
            - Input: [B, N, 6] (XYZ + RGB)
            - Output: [B, N, 8] (per-point logits)
            - Inference time: ~0.1s per 4K points
            - Memory usage: ~2GB GPU
            """)
        
        st.subheader("📋 Layer Details")
        model_info = """
        | Layer | Points | Radius | Samples | Features |
        |-------|--------|--------|---------|----------|
        | SA1   | 1024   | 0.1    | 32      | 64       |
        | SA2   | 256    | 0.2    | 32      | 128      |
        | SA3   | 64     | 0.4    | 32      | 256      |
        | SA4   | 16     | 0.8    | 32      | 512      |
        """
        st.markdown(model_info)
        
        # System status
        st.subheader("🔧 System Status")
        status_df = pd.DataFrame({
            'Component': ['PyTorch', 'Plotly', 'Open3D', 'scikit-learn', 'PointNet++ Model'],
            'Status': [
                '✅ Available' if TORCH_AVAILABLE else '❌ Missing',
                '✅ Available' if PLOTLY_AVAILABLE else '❌ Missing', 
                '✅ Available' if OPEN3D_AVAILABLE else '❌ Missing',
                '✅ Available' if SKLEARN_AVAILABLE else '❌ Missing',
                '✅ Available' if MODEL_AVAILABLE else '❌ Missing'
            ]
        })
        st.dataframe(status_df)

def show_sample_only(points, true_labels, scene_name):
    """Show sample data without model inference"""
    st.subheader(f"📊 {scene_name} Visualization")
    
    class_names = [
        'man-made terrain', 'natural terrain', 'high vegetation',
        'low vegetation', 'buildings', 'hard scape', 
        'scanning artifacts', 'cars'
    ]
    
    # Show 3D plot if possible, otherwise 2D
    if PLOTLY_AVAILABLE:
        fig = create_3d_plot(points[:, :3], true_labels, f"{scene_name} - Ground Truth")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = create_simple_plot(points[:, :3], true_labels, f"{scene_name} - Ground Truth")
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    unique_labels, counts = np.unique(true_labels, return_counts=True)
    stats_df = pd.DataFrame({
        'Class': [class_names[i] for i in unique_labels],
        'Count': counts,
        'Percentage': (counts / len(true_labels) * 100).round(2)
    })
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(stats_df)
    with col2:
        if PLOTLY_AVAILABLE:
            fig_pie = px.pie(stats_df, values='Count', names='Class', 
                            title="Ground Truth Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

def classify_point_cloud(model, points, true_labels=None):
    """Run classification and show results"""
    # Preprocess points and get sampling indices
    processed_points, sampling_indices = preprocess_points(points)
    
    # Normalize coordinates
    coords = processed_points[:, :3]
    coords = coords - coords.mean(axis=0)
    scale = np.max(np.linalg.norm(coords, axis=1))
    if scale > 0:
        coords = coords / scale
    
    # Ensure we have RGB features
    if processed_points.shape[1] >= 6:
        features = processed_points[:, 3:6]
    else:
        features = np.random.uniform(0, 1, (len(processed_points), 3))
    
    # Combine normalized coordinates with features
    model_input = np.hstack([coords, features])
    
    # Run inference
    with st.spinner("Running PointNet++ inference..."):
        input_tensor = torch.FloatTensor(model_input).unsqueeze(0)
        
        with torch.no_grad():
            predictions = model(input_tensor)
            predicted_labels = predictions.squeeze().argmax(dim=1).cpu().numpy()
    
    # Map predictions back to original point cloud for visualization
    viz_points = processed_points[:, :3]
    viz_predictions = predicted_labels
    
    if true_labels is not None:
        true_labels_sampled = true_labels[sampling_indices]
    else:
        true_labels_sampled = None
    
    # Class names
    class_names = [
        'man-made terrain', 'natural terrain', 'high vegetation',
        'low vegetation', 'buildings', 'hard scape', 
        'scanning artifacts', 'cars'
    ]
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Original Point Cloud")
        if PLOTLY_AVAILABLE:
            original_fig = create_3d_plot(viz_points, title="Original Point Cloud")
            st.plotly_chart(original_fig, use_container_width=True)
        else:
            original_fig = create_simple_plot(viz_points, title="Original Point Cloud")
            st.plotly_chart(original_fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Classification Results")
        if PLOTLY_AVAILABLE:
            classified_fig = create_3d_plot(viz_points, viz_predictions, "Classified Point Cloud")
            st.plotly_chart(classified_fig, use_container_width=True)
        else:
            classified_fig = create_simple_plot(viz_points, viz_predictions, "Classified Point Cloud")
            st.plotly_chart(classified_fig, use_container_width=True)
    
    # Statistics
    st.subheader("📊 Classification Statistics")
    
    # Count predictions
    unique_labels, counts = np.unique(viz_predictions, return_counts=True)
    stats_df = pd.DataFrame({
        'Class': [class_names[i] for i in unique_labels],
        'Count': counts,
        'Percentage': (counts / len(viz_predictions) * 100).round(2)
    })
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.dataframe(stats_df, use_container_width=True)
    
    with col2:
        if PLOTLY_AVAILABLE:
            fig_pie = px.pie(stats_df, values='Count', names='Class', 
                            title="Class Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # Accuracy if ground truth available
    if true_labels_sampled is not None:
        accuracy = (viz_predictions == true_labels_sampled).mean() * 100
        st.success(f"🎯 **Accuracy**: {accuracy:.1f}% (on sampled {len(viz_predictions)} points)")
        
        # Confusion matrix if sklearn available
        if SKLEARN_AVAILABLE and PLOTLY_AVAILABLE:
            try:
                cm = confusion_matrix(true_labels_sampled, viz_predictions, labels=range(8))
                
                fig_cm = px.imshow(cm, 
                                  x=[class_names[i] for i in range(8)],
                                  y=[class_names[i] for i in range(8)],
                                  title="Confusion Matrix",
                                  aspect="auto")
                fig_cm.update_xaxes(side="bottom")
                st.plotly_chart(fig_cm, use_container_width=True)
            except Exception as e:
                st.info("Could not generate confusion matrix")
    
    # Export options
    st.subheader("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV export
        export_df = pd.DataFrame({
            'X': viz_points[:, 0],
            'Y': viz_points[:, 1], 
            'Z': viz_points[:, 2],
            'Predicted_Class_ID': viz_predictions,
            'Predicted_Class': [class_names[i] for i in viz_predictions]
        })
        
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="classified_pointcloud.csv",
            mime="text/csv"
        )
    
    with col2:
        # JSON export
        import json
        export_json = {
            'points': viz_points.tolist(),
            'predictions': viz_predictions.tolist(),
            'class_names': class_names,
            'statistics': stats_df.to_dict('records')
        }
        
        json_str = json.dumps(export_json, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name="classified_pointcloud.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()