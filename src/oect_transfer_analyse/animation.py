# animation.py
"""
Advanced animation generation for OECT transfer curves

This module provides enhanced animation capabilities for creating videos
of transfer curve evolution and parameter changes over time.
"""

import os
import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional
from functools import partial
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Import from core package
try:
    import oect_transfer as ot
except ImportError:
    raise ImportError("oect-transfer is required. Install with: pip install oect-transfer")

# Handle optional dependencies
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for performance
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


def _check_dependencies():
    """Check if required dependencies are available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Matplotlib is required for animation functionality. "
            "Install it with: pip install matplotlib"
        )
    if not OPENCV_AVAILABLE:
        raise ImportError(
            "OpenCV is required for video generation. "
            "Install it with: pip install opencv-python"
        )


def _generate_single_frame(
    frame_data: Tuple[int, Dict[str, Any]], 
    xlim: Tuple[float, float],
    ylim_linear: Tuple[float, float], 
    ylim_log: Tuple[float, float],
    figsize: Tuple[float, float], 
    dpi: int,
    show_frame_info: bool = True,
    style: str = 'standard'
) -> np.ndarray:
    """
    Generate a single frame for animation with enhanced styling.
    
    Parameters
    ----------
    frame_data : Tuple[int, Dict[str, Any]]
        Frame index and transfer object data
    xlim : Tuple[float, float]
        X-axis limits
    ylim_linear : Tuple[float, float]
        Y-axis limits for linear scale
    ylim_log : Tuple[float, float]
        Y-axis limits for log scale
    figsize : Tuple[float, float]
        Figure size
    dpi : int
        Figure resolution
    show_frame_info : bool, default True
        Whether to show frame information
    style : str, default 'standard'
        Plot style: 'standard', 'publication', 'minimal'
    
    Returns
    -------
    np.ndarray
        RGB image array for the frame
    """
    # Ensure correct backend in subprocess
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    frame_idx, transfer_obj = frame_data
    
    # Style-specific settings
    if style == 'publication':
        plt.rcParams.update({
            'font.size': 12,
            'axes.linewidth': 1.5,
            'xtick.major.width': 1.5,
            'ytick.major.width': 1.5,
        })
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    
    # Left subplot - Linear scale
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("$V_{GS}$ (V)", fontsize=12)
    ax1.set_ylabel("$|I_{DS}|$ (A)", fontsize=12)
    ax1.set_title("Linear Scale", fontsize=14)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim_linear)
    
    # Right subplot - Log scale
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("$V_{GS}$ (V)", fontsize=12)
    ax2.set_ylabel("$|I_{DS}|$ (A)", fontsize=12)
    ax2.set_title("Log Scale", fontsize=14)
    ax2.set_yscale('log')
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim_log)
    
    # Plot data
    try:
        transfer = transfer_obj['transfer']
        vg_data = transfer.Vg.raw
        id_data = transfer.I.raw
        
        # Enhanced styling
        line_kwargs = {
            'linewidth': 3 if style == 'publication' else 2,
            'alpha': 0.9
        }
        
        # Left subplot - Linear scale
        ax1.plot(vg_data, np.abs(id_data), color='tab:blue', **line_kwargs)
        
        # Right subplot - Log scale
        id_abs = np.abs(id_data)
        valid_mask = id_abs > 0
        if np.any(valid_mask):
            ax2.plot(vg_data[valid_mask], id_abs[valid_mask], 
                    color='tab:red', **line_kwargs)
        
        # Add analysis markers for publication style
        if style == 'publication':
            try:
                von_voltage = transfer.Von.raw
                von_current = np.interp(von_voltage, vg_data, id_abs)
                
                ax1.plot(von_voltage, von_current, 'go', markersize=8, label='$V_{on}$')
                ax2.plot(von_voltage, von_current, 'go', markersize=8, label='$V_{on}$')
                
                ax1.legend(fontsize=10)
                ax2.legend(fontsize=10)
            except:
                pass
        
        # Set frame title
        if show_frame_info:
            filename = transfer_obj.get('filename', f'Frame_{frame_idx}')
            title_text = f"Frame {frame_idx+1}: {filename}"
            if style == 'minimal':
                title_text = f"Measurement {frame_idx+1}"
            
            fig.suptitle(title_text, fontsize=16 if style == 'publication' else 14, 
                        y=0.95)
    
    except Exception as e:
        print(f"Warning: Error plotting frame {frame_idx}: {e}")
        # Plot empty frame with error message
        for ax in [ax1, ax2]:
            ax.text(0.5, 0.5, f"Error in frame {frame_idx}", 
                   transform=ax.transAxes, ha='center', va='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Convert to numpy array
    fig.canvas.draw()
    try:
        buf = fig.canvas.buffer_rgba()
        buf = np.asarray(buf)
        buf = buf[:, :, :3]  # Convert RGBA to RGB
    except AttributeError:
        try:
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except AttributeError:
            import io
            try:
                from PIL import Image
                buf_io = io.BytesIO()
                fig.savefig(buf_io, format='png', dpi=dpi, bbox_inches='tight')
                buf_io.seek(0)
                img = Image.open(buf_io)
                buf = np.array(img)
                if buf.shape[2] == 4:
                    buf = buf[:, :, :3]
            except ImportError:
                raise ImportError("PIL is required for animation. Install with: pip install Pillow")
    
    plt.close(fig)
    return buf


def generate_transfer_animation(
    transfer_objects: List[Dict[str, Any]],
    output_path: str = "transfer_evolution.mp4",
    fps: int = 30,
    dpi: int = 100,
    xlim: Optional[Tuple[float, float]] = None,
    ylim_linear: Optional[Tuple[float, float]] = None,
    ylim_log: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (12, 5),
    n_workers: Optional[int] = None,
    codec: str = 'mp4v',
    show_frame_info: bool = True,
    style: str = 'standard',
    method: str = 'parallel'
) -> None:
    """
    Generate enhanced transfer curve evolution animation.
    
    Parameters
    ----------
    transfer_objects : List[Dict[str, Any]]
        List of transfer objects
    output_path : str, default "transfer_evolution.mp4"
        Output video file path
    fps : int, default 30
        Frames per second
    dpi : int, default 100
        Frame resolution
    xlim : Tuple[float, float], optional
        X-axis limits
    ylim_linear : Tuple[float, float], optional
        Y-axis limits for linear scale
    ylim_log : Tuple[float, float], optional
        Y-axis limits for log scale
    figsize : Tuple[float, float], default (12, 5)
        Figure size
    n_workers : int, optional
        Number of parallel workers
    codec : str, default 'mp4v'
        Video codec
    show_frame_info : bool, default True
        Whether to show frame information
    style : str, default 'standard'
        Animation style: 'standard', 'publication', 'minimal'
    method : str, default 'parallel'
        Generation method: 'parallel' or 'memory'
    
    Examples
    --------
    >>> # High-quality publication animation
    >>> generate_transfer_animation(
    ...     transfer_objects,
    ...     output_path='device_evolution.mp4',
    ...     style='publication',
    ...     fps=24,
    ...     dpi=150
    ... )
    """
    _check_dependencies()
    
    if not transfer_objects:
        raise ValueError("transfer_objects list is empty")
    
    print(f"🎬 Generating {style} style animation with {len(transfer_objects)} frames...")
    start_time = time.time()
    
    # Auto-determine coordinate ranges
    if xlim is None:
        all_vg = np.concatenate([item['transfer'].Vg.raw for item in transfer_objects])
        xlim = (np.min(all_vg), np.max(all_vg))
    
    if ylim_linear is None:
        all_id = np.concatenate([np.abs(item['transfer'].I.raw) for item in transfer_objects])
        ylim_linear = (np.min(all_id), np.max(all_id))
    
    if ylim_log is None:
        all_id_abs = []
        for item in transfer_objects:
            id_data = np.abs(item['transfer'].I.raw)
            id_data = id_data[id_data > 0]
            if len(id_data) > 0:
                all_id_abs.extend(id_data)
        
        if len(all_id_abs) > 0:
            all_id_abs = np.array(all_id_abs)
            ylim_log = (np.min(all_id_abs) * 0.1, np.max(all_id_abs) * 10)
        else:
            ylim_log = (1e-12, 1e-3)
    
    # Prepare frame generation data
    frame_data_list = [(i, obj) for i, obj in enumerate(transfer_objects)]
    
    # Determine number of workers
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(transfer_objects))
    
    print(f"Using {n_workers} processes for frame generation...")
    
    # Generate frames
    generate_frame_func = partial(
        _generate_single_frame,
        xlim=xlim,
        ylim_linear=ylim_linear,
        ylim_log=ylim_log,
        figsize=figsize,
        dpi=dpi,
        show_frame_info=show_frame_info,
        style=style
    )
    
    if method == 'parallel':
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            frames = list(executor.map(generate_frame_func, frame_data_list))
    else:  # memory method
        frames = []
        for frame_data in frame_data_list:
            frames.append(generate_frame_func(frame_data))
    
    frame_gen_time = time.time()
    print(f"Frame generation completed in {frame_gen_time - start_time:.2f}s")
    
    # Write video
    if len(frames) > 0:
        _write_video_enhanced(frames, output_path, fps, codec)
    
    total_time = time.time() - start_time
    print(f"🎉 Animation completed! Total time: {total_time:.2f}s")
    print(f"📁 Video saved as: {output_path}")


def create_animation_preview(
    transfer_objects: List[Dict[str, Any]],
    indices: List[int],
    output_path: str = "preview.png",
    figsize: Tuple[float, float] = (16, 10),
    dpi: int = 150,
    style: str = 'standard'
) -> None:
    """
    Create enhanced animation preview with multiple frames.
    
    Parameters
    ----------
    transfer_objects : List[Dict[str, Any]]
        List of transfer objects
    indices : List[int]
        Indices of frames to include
    output_path : str, default "preview.png"
        Output image path
    figsize : Tuple[float, float], default (16, 10)
        Figure size
    dpi : int, default 150
        Image resolution
    style : str, default 'standard'
        Preview style
    """
    _check_dependencies()
    
    if not indices:
        raise ValueError("indices list is empty")
    
    # Validate indices
    max_idx = len(transfer_objects) - 1
    validated_indices = []
    for idx in indices:
        if idx < 0:
            idx = len(transfer_objects) + idx
        if 0 <= idx <= max_idx:
            validated_indices.append(idx)
        else:
            print(f"Warning: Index {idx} out of range, skipping")
    
    if not validated_indices:
        raise ValueError("No valid indices provided")
    
    n_frames = len(validated_indices)
    
    # Create subplot grid
    if n_frames <= 3:
        rows, cols = 2, n_frames
    elif n_frames <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, (n_frames + 2) // 3
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
    if n_frames == 1:
        axes = np.array([axes])
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Style settings
    if style == 'publication':
        plt.rcParams.update({'font.size': 10})
    
    # Auto-determine coordinate ranges
    all_vg = np.concatenate([transfer_objects[i]['transfer'].Vg.raw for i in validated_indices])
    xlim = (np.min(all_vg), np.max(all_vg))
    
    all_id = np.concatenate([np.abs(transfer_objects[i]['transfer'].I.raw) for i in validated_indices])
    ylim_linear = (np.min(all_id), np.max(all_id))
    
    all_id_abs = []
    for i in validated_indices:
        id_data = np.abs(transfer_objects[i]['transfer'].I.raw)
        id_data = id_data[id_data > 0]
        if len(id_data) > 0:
            all_id_abs.extend(id_data)
    
    if len(all_id_abs) > 0:
        all_id_abs = np.array(all_id_abs)
        ylim_log = (np.min(all_id_abs) * 0.1, np.max(all_id_abs) * 10)
    else:
        ylim_log = (1e-12, 1e-3)
    
    # Plot each frame
    for i, idx in enumerate(validated_indices):
        if i >= len(axes):
            break
            
        ax = axes[i]
        transfer_obj = transfer_objects[idx]
        transfer = transfer_obj['transfer']
        vg_data = transfer.Vg.raw
        id_data = np.abs(transfer.I.raw)
        
        # Use log scale for preview
        valid_mask = id_data > 0
        if np.any(valid_mask):
            ax.semilogy(vg_data[valid_mask], id_data[valid_mask], 
                       linewidth=2, color='tab:blue', alpha=0.8)
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim_log)
        ax.grid(True, alpha=0.3)
        
        # Labels and title
        filename = transfer_obj.get('filename', f'Frame_{idx}')
        if style == 'minimal':
            title = f"#{idx+1}"
        else:
            title = f"Frame {idx+1}"
        
        ax.set_title(title, fontsize=12)
        
        if i // cols == rows - 1:  # Bottom row
            ax.set_xlabel('$V_{GS}$ (V)', fontsize=10)
        if i % cols == 0:  # Left column
            ax.set_ylabel('$|I_{DS}|$ (A)', fontsize=10)
    
    # Hide unused subplots
    for i in range(len(validated_indices), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Transfer Curve Evolution Preview ({len(validated_indices)} frames)', 
                fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"🖼️  Preview image saved as: {output_path}")
    plt.show()


def create_parameter_animation(
    extractor,
    output_path: str = "parameter_evolution.mp4",
    parameters: Optional[List[str]] = None,
    fps: int = 30,
    duration: float = 10.0,
    style: str = 'standard'
) -> None:
    """
    Create animation showing parameter evolution over time.
    
    Parameters
    ----------
    extractor : TransferTimeSeriesExtractor
        Configured extractor with time-series data
    output_path : str, default "parameter_evolution.mp4"
        Output video path
    parameters : List[str], optional
        Parameters to animate
    fps : int, default 30
        Frames per second
    duration : float, default 10.0
        Animation duration in seconds
    style : str, default 'standard'
        Animation style
    """
    _check_dependencies()
    
    if extractor.time_series_data is None:
        extractor.extract_time_series()
    
    if parameters is None:
        parameters = ['gm_max_raw', 'Von_raw', 'I_max_raw']
    
    print(f"🎬 Creating parameter evolution animation...")
    
    # This would be implemented with matplotlib animation
    # For now, create a static plot
    fig, axes = plt.subplots(len(parameters), 1, figsize=(10, 8), sharex=True)
    if len(parameters) == 1:
        axes = [axes]
    
    time_points = extractor.time_series_data.time_points
    
    for i, param in enumerate(parameters):
        param_data = getattr(extractor.time_series_data, param)
        axes[i].plot(time_points, param_data, 'o-', linewidth=2, markersize=4)
        axes[i].set_ylabel(param, fontsize=12)
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time Point', fontsize=12)
    plt.title('Parameter Evolution Over Time', fontsize=14)
    plt.tight_layout()
    
    # Save as static image for now
    static_path = output_path.replace('.mp4', '.png')
    plt.savefig(static_path, dpi=150, bbox_inches='tight')
    print(f"📊 Parameter plot saved as: {static_path}")
    print("Note: Full parameter animation requires additional development")
    plt.show()


def batch_animation_generation(
    data_folders: List[str],
    output_dir: str = "batch_animations",
    device_type: str = "N",
    **animation_kwargs
) -> List[str]:
    """
    Generate animations for multiple datasets in batch.
    
    Parameters
    ----------
    data_folders : List[str]
        List of data folder paths
    output_dir : str, default "batch_animations"
        Output directory for animations
    device_type : str, default "N"
        Device type
    **animation_kwargs
        Additional arguments for generate_transfer_animation
        
    Returns
    -------
    List[str]
        List of generated animation paths
    """
    _check_dependencies()
    
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    
    print(f"🎬 Batch animation generation for {len(data_folders)} datasets...")
    
    for i, folder in enumerate(data_folders):
        try:
            # Load data
            transfer_objects = ot.load_all_transfer_files(folder, device_type)
            
            # Generate animation
            folder_name = os.path.basename(folder.rstrip('/'))
            output_path = os.path.join(output_dir, f"{folder_name}_evolution.mp4")
            
            print(f"Processing {i+1}/{len(data_folders)}: {folder_name}")
            
            generate_transfer_animation(
                transfer_objects,
                output_path=output_path,
                **animation_kwargs
            )
            
            generated_files.append(output_path)
            
        except Exception as e:
            print(f"Error processing {folder}: {e}")
    
    print(f"🎉 Batch generation completed! {len(generated_files)} animations created.")
    return generated_files


def _write_video_enhanced(frames: List[np.ndarray], 
                         output_path: str, 
                         fps: int, 
                         codec: str) -> None:
    """Enhanced video writing with better error handling."""
    if not frames:
        raise ValueError("No frames to write")
    
    height, width, channels = frames[0].shape
    
    # Try different codecs if the specified one fails
    codecs_to_try = [codec, 'mp4v', 'XVID', 'MJPG']
    
    for codec_name in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                continue
            
            print(f"Writing video with {codec_name} codec...")
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            # Verify file was created
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"✅ Video successfully written with {codec_name} codec")
                return
            
        except Exception as e:
            print(f"Failed with {codec_name}: {e}")
            continue
    
    raise RuntimeError("Failed to write video with any available codec")