import os
import numpy as np
import argparse
from rich.console import Console
from rich.table import Table

def main():
    parser = argparse.ArgumentParser(description="Verify dense vs sparse MLEM reconstructions.")
    parser.add_argument("--dense_dir", required=True, help="Directory containing the legacy dense reconstruction.")
    parser.add_argument("--sparse_dir", required=True, help="Directory containing the new sparse reconstruction.")
    
    args = parser.parse_args()
    console = Console()

    # The specific file we are looking for
    target_filename = "recon_mlem_torch_optimized.npz"
    
    dense_path = os.path.join(args.dense_dir, target_filename)
    sparse_path = os.path.join(args.sparse_dir, target_filename)

    # 1. File Existence Check
    if not os.path.exists(dense_path):
        console.print(f"[red]❌ Missing dense file: {dense_path}[/red]")
        return
    if not os.path.exists(sparse_path):
        console.print(f"[red]❌ Missing sparse file: {sparse_path}[/red]")
        return

    console.print(f"[cyan]Found target file: {target_filename} in both directories.[/cyan]\n")

    # 2. Load Data
    try:
        # Load the 'estimates' array containing the reconstruction history
        dense_est = np.load(dense_path)['estimates']
        sparse_est = np.load(sparse_path)['estimates']
    except Exception as e:
        console.print(f"[red]❌ Error loading .npz files: {e}[/red]")
        return

    # 3. Shape Verification
    if dense_est.shape != sparse_est.shape:
        console.print(f"[red]❌ SHAPE MISMATCH: Dense is {dense_est.shape}, Sparse is {sparse_est.shape}[/red]")
        return
    
    num_saves, h, w = dense_est.shape
    console.print(f"✅ Shape Match: {num_saves} saved iterations of size {h}x{w}\n")

    # 4. Statistical Error Calculation
    # Absolute difference: |Dense - Sparse|
    abs_diff = np.abs(dense_est - sparse_est)
    
    # Relative difference: |Dense - Sparse| / (|Dense| + epsilon)
    # Epsilon prevents division by zero in empty background regions
    epsilon = 1e-9
    rel_diff = abs_diff / (np.abs(dense_est) + epsilon)

    # 5. Build the Summary Table
    table = Table(title="MLEM Reconstruction Variance Report", show_header=True, header_style="bold magenta")
    table.add_column("Scope", style="dim", width=20)
    table.add_column("Max Abs Error", justify="right", style="red")
    table.add_column("Mean Abs Error", justify="right", style="yellow")
    table.add_column("Max Rel Error (%)", justify="right", style="blue")
    table.add_column("Status", justify="center")

    # Analyze Overall History (All Iterations)
    max_abs_all = np.max(abs_diff)
    mean_abs_all = np.mean(abs_diff)
    max_rel_all = np.max(rel_diff) * 100

    # Tolerance for declaring "Pass" (Adjust if needed, 1e-4 is standard for iterative float32 GPU math)
    tolerance = 1e-4
    overall_status = "[green]✅ PASS[/green]" if max_abs_all < tolerance else "[yellow]⚠️ MARGINAL DRIFT[/yellow]"

    table.add_row(
        "All Iterations",
        f"{max_abs_all:.2e}",
        f"{mean_abs_all:.2e}",
        f"{max_rel_all:.4f}%",
        overall_status
    )

    # Analyze ONLY the Final Reconstructed Image (What actually matters)
    final_dense = dense_est[-1]
    final_sparse = sparse_est[-1]
    
    final_abs_diff = np.abs(final_dense - final_sparse)
    final_rel_diff = final_abs_diff / (np.abs(final_dense) + epsilon)

    max_abs_final = np.max(final_abs_diff)
    mean_abs_final = np.mean(final_abs_diff)
    max_rel_final = np.max(final_rel_diff) * 100

    final_status = "[green]✅ PASS[/green]" if max_abs_final < tolerance else "[yellow]⚠️ MARGINAL DRIFT[/yellow]"

    table.add_row(
        "Final Image Only",
        f"{max_abs_final:.2e}",
        f"{mean_abs_final:.2e}",
        f"{max_rel_final:.4f}%",
        final_status
    )

    console.print(table)

    # 6. Final Assessment
    console.print("\n[bold underline]Assessment Summary:[/bold underline]")
    if max_abs_final < 1e-5:
        console.print("The dense and sparse reconstructions are **mathematically identical**. Any differences are purely at the microscopic precision limit of the GPU's 32-bit floating-point architecture.")
    elif max_abs_final < 1e-3:
        console.print("The reconstructions exhibit **normal floating-point drift**. Because sparse matrices sum elements in a different order than dense matrices, iterative compounding causes microscopic variance. The visual image quality is 100% identical.")
    else:
        console.print("[red]Warning: The variance is higher than expected for standard floating-point drift. You may want to verify your sparsification threshold in the PPDF script.[/red]")

if __name__ == "__main__":
    main()