"""Animation generation for OECT transfer curve evolution."""

import os
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

# Check for required dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from oect_transfer import Transfer
except ImportError:
    raise ImportError(
        "oect-transfer package is required. Install with: pip install oect-transfer"
    )


def _check_animation_dependencies():
    """Check if animation dependencies are available."""
    missing = []
    if not CV2_AVAILABLE:
        missing.append("opencv-python")
    if not PIL_AVAILABLE:
        missing.append("Pillow")
    if not MATPLOTLIB_AVAILABLE:
        missing.append("matplotlib")
    
    if missing:
        raise ImportError(
            f"Animation requires additional dependencies: {', '.join(missing)}. "
            f"Install with: pip install oect-transfer-analysis[animation]"
        )


def generate_single_frame(
    frame_data: Tuple[int, Dict[str, Any]],
    xlim: Tuple[float, float],
    ylim_linear: Tuple[float, float],
    ylim_log: Tuple[float, float],
    figsize: Tuple[float, float],
    dpi: int
) -> np.ndarray:
    """
    Generate a single frame for animation (used in parallel processing).
    
    Parameters
    ----------
    frame_data : Tuple[int, Dict[str, Any]]
        Tuple of (frame_index, transfer_object)
    xlim : Tuple[float, float]
        X-axis limits
    ylim_linear : Tuple[float, float]
        Y-axis limits for linear scale
    ylim_log : Tuple[float, float]
        Y-axis limits for log scale
    figsize : Tuple[float, float]
        Figure size
    dpi : int
        Figure DPI
        
    Returns
    -------
    np.ndarray
        Frame image as numpy array
    """
    # Ensure correct backend in subprocess
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    frame_idx, transfer_obj = frame_data
    
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
    
    # Get data
    transfer = transfer_obj['transfer']
    vg_data = transfer.Vg.raw
    id_data = transfer.I.raw
    id_abs = np.abs(id_data)
    
    # Plot linear scale
    ax1.plot(vg_data, id_abs, linewidth=2, color='tab:blue')
    
    # Plot log scale
    valid_mask = id_abs > 0
    if np.any(valid_mask):
        ax2.plot(vg_data[valid_mask], id_abs[valid_mask], linewidth=2, color='tab:red')
    
    # Set main title
    filename = transfer_obj['filename']
    fig.suptitle(f"Frame {frame_idx+1}: {filename}", fontsize=16)
    
    plt.tight_layout()
    
    # Convert to numpy array
    fig.canvas.draw()
    try:
        # Try new matplotlib API
        buf = fig.canvas.buffer_rgba()
        buf = np.asarray(buf)
        buf = buf[:, :, :3]  # Convert RGBA to RGB
    except AttributeError:
        try:
            # Try older API
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except AttributeError:
            # Fallback to file-based method
            import io
            buf_io = io.BytesIO()
            fig.savefig(buf_io, format='png', dpi=dpi, bbox_inches='tight')
            buf_io.seek(0)
            
            if PIL_AVAILABLE:
                img = Image.open(buf_io)
                buf = np.array(img)
                if buf.shape[2] == 4:  # RGBA
                    buf = buf[:, :, :3]  # Convert to RGB
            else:
                raise RuntimeError("Cannot convert figure to array without PIL")
    
    plt.close(fig)
    return buf


class AnimationGenerator:
    """Generator for OECT transfer curve evolution animations."""
    
    def __init__(self):
        """Initialize the animation generator."""
        _check_animation_dependencies()
    
    def generate_animation(
        self,
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
        verbose: bool = True
    ) -> str:
        """
        Generate transfer curve evolution animation.
        
        Parameters
        ----------
        transfer_objects : List[Dict[str, Any]]
            List of transfer objects
        output_path : str, default "transfer_evolution.mp4"
            Output video file path
        fps : int, default 30
            Frames per second
        dpi : int, default 100
            Figure DPI (lower = faster generation)
        xlim : tuple, optional
            X-axis limits. If None, auto-calculated
        ylim_linear : tuple, optional
            Y-axis limits for linear scale. If None, auto-calculated
        ylim_log : tuple, optional
            Y-axis limits for log scale. If None, auto-calculated
        figsize : tuple, default (12, 5)
            Figure size in inches
        n_workers : int, optional
            Number of parallel workers. If None, uses CPU count
        codec : str, default 'mp4v'
            Video codec ('mp4v', 'XVID', 'H264')
        verbose : bool, default True
            Whether to print progress
            
        Returns
        -------
        str
            Path to generated video file
        """
        if not transfer_objects:
            raise ValueError("transfer_objects list is empty")
        
        if verbose:
            print(f"Generating animation with {len(transfer_objects)} frames...")
        
        start_time = time.time()
        
        # Auto-determine coordinate ranges
        xlim = xlim or self._calculate_xlim(transfer_objects)
        ylim_linear = ylim_linear or self._calculate_ylim_linear(transfer_objects)
        ylim_log = ylim_log or self._calculate_ylim_log(transfer_objects)
        
        if verbose:
            print(f"X-axis range: {xlim}")
            print(f"Y-axis range (linear): {ylim_linear}")
            print(f"Y-axis range (log): {ylim_log}")
        
        # Prepare parallel processing
        frame_data_list = [(i, obj) for i, obj in enumerate(transfer_objects)]
        
        if n_workers is None:
            n_workers = min(mp.cpu_count(), len(transfer_objects))
        
        if verbose:
            print(f"Using {n_workers} workers for parallel frame generation...")
        
        # Generate frames in parallel
        generate_frame_func = partial(
            generate_single_frame,
            xlim=xlim,
            ylim_linear=ylim_linear,
            ylim_log=ylim_log,
            figsize=figsize,
            dpi=dpi
        )
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            frames = list(executor.map(generate_frame_func, frame_data_list))
        
        frame_gen_time = time.time()
        if verbose:
            print(f"Frame generation completed in {frame_gen_time - start_time:.2f}s")
        
        # Write video
        if frames:
            self._write_video(frames, output_path, fps, codec, verbose)
        else:
            raise RuntimeError("No frames were generated")
        
        total_time = time.time() - start_time
        if verbose:
            print(f"Animation generation completed in {total_time:.2f}s")
            print(f"Average time per frame: {total_time/len(transfer_objects):.3f}s")
            print(f"Video saved to: {output_path}")
        
        return output_path
    
    def generate_memory_optimized(
        self,
        transfer_objects: List[Dict[str, Any]],
        output_path: str = "transfer_evolution_memory.mp4",
        batch_size: int = 50,
        **kwargs
    ) -> str:
        """
        Generate animation with memory optimization for large datasets.
        
        Parameters
        ----------
        transfer_objects : List[Dict[str, Any]]
            List of transfer objects
        output_path : str, default "transfer_evolution_memory.mp4"
            Output video file path
        batch_size : int, default 50
            Number of frames to process in each batch
        **kwargs
            Additional arguments passed to generate_animation()
            
        Returns
        -------
        str
            Path to generated video file
        """
        verbose = kwargs.get('verbose', True)
        
        if verbose:
            print(f"Generating memory-optimized animation with batch size {batch_size}...")
        
        # Set defaults
        kwargs.setdefault('fps', 30)
        kwargs.setdefault('dpi', 100)
        kwargs.setdefault('figsize', (12, 5))
        kwargs.setdefault('codec', 'mp4v')
        
        start_time = time.time()
        
        # Calculate coordinate ranges
        xlim = kwargs.get('xlim') or self._calculate_xlim(transfer_objects)
        ylim_linear = kwargs.get('ylim_linear') or self._calculate_ylim_linear(transfer_objects)
        ylim_log = kwargs.get('ylim_log') or self._calculate_ylim_log(transfer_objects)
        
        # Generate first frame to get dimensions
        sample_frame = generate_single_frame(
            (0, transfer_objects[0]), xlim, ylim_linear, ylim_log,
            kwargs['figsize'], kwargs['dpi']
        )
        height, width, channels = sample_frame.shape
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*kwargs['codec'])
        out = cv2.VideoWriter(output_path, fourcc, kwargs['fps'], (width, height))
        
        # Write first frame
        frame_bgr = cv2.cvtColor(sample_frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        
        # Process remaining frames in batches
        total_frames = len(transfer_objects)
        for batch_start in range(1, total_frames, batch_size):
            batch_end = min(batch_start + batch_size, total_frames)
            
            if verbose:
                print(f"Processing batch {batch_start}-{batch_end-1}/{total_frames}")
            
            # Prepare batch data
            batch_data = [(i, transfer_objects[i]) for i in range(batch_start, batch_end)]
            
            # Generate batch frames
            generate_frame_func = partial(
                generate_single_frame,
                xlim=xlim,
                ylim_linear=ylim_linear,
                ylim_log=ylim_log,
                figsize=kwargs['figsize'],
                dpi=kwargs['dpi']
            )
            
            with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
                batch_frames = list(executor.map(generate_frame_func, batch_data))
            
            # Write batch frames
            for frame in batch_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
        
        out.release()
        
        total_time = time.time() - start_time
        if verbose:
            print(f"Memory-optimized animation completed in {total_time:.2f}s")
            print(f"Video saved to: {output_path}")
        
        return output_path
    
    def _calculate_xlim(self, transfer_objects: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Calculate appropriate X-axis limits."""
        all_vg = np.concatenate([obj['transfer'].Vg.raw for obj in transfer_objects])
        return (np.min(all_vg), np.max(all_vg))
    
    def _calculate_ylim_linear(self, transfer_objects: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Calculate appropriate Y-axis limits for linear scale."""
        all_id = np.concatenate([np.abs(obj['transfer'].I.raw) for obj in transfer_objects])
        return (np.min(all_id), np.max(all_id))
    
    def _calculate_ylim_log(self, transfer_objects: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Calculate appropriate Y-axis limits for log scale."""
        all_id_abs = []
        for obj in transfer_objects:
            id_data = np.abs(obj['transfer'].I.raw)
            id_data = id_data[id_data > 0]
            if len(id_data) > 0:
                all_id_abs.extend(id_data)
        
        if len(all_id_abs) > 0:
            all_id_abs = np.array(all_id_abs)
            return (np.min(all_id_abs) * 0.1, np.max(all_id_abs) * 10)
        else:
            return (1e-12, 1e-3)
    
    def _write_video(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: int,
        codec: str,
        verbose: bool = True
    ) -> None:
        """Write frames to video file."""
        if not frames:
            raise ValueError("No frames to write")
        
        height, width, channels = frames[0].shape
        
        # Setup video codec
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if verbose:
            print("Writing video...")
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()


# Convenience functions
def generate_transfer_animation(
    transfer_objects: List[Dict[str, Any]],
    output_path: str = "transfer_evolution.mp4",
    **kwargs
) -> str:
    """
    Convenience function to generate transfer animation.
    
    Parameters
    ----------
    transfer_objects : List[Dict[str, Any]]
        List of transfer objects
    output_path : str, default "transfer_evolution.mp4"
        Output video file path
    **kwargs
        Additional arguments passed to AnimationGenerator.generate_animation()
        
    Returns
    -------
    str
        Path to generated video file
    """
    generator = AnimationGenerator()
    return generator.generate_animation(transfer_objects, output_path, **kwargs)


def generate_transfer_animation_optimized(
    transfer_objects: List[Dict[str, Any]],
    output_path: str = "transfer_evolution_optimized.mp4",
    **kwargs
) -> str:
    """
    Convenience function to generate optimized transfer animation.
    
    This is an alias for generate_transfer_animation with optimized defaults.
    """
    # Set optimized defaults
    kwargs.setdefault('dpi', 100)
    kwargs.setdefault('n_workers', mp.cpu_count())
    kwargs.setdefault('codec', 'mp4v')
    
    return generate_transfer_animation(transfer_objects, output_path, **kwargs)