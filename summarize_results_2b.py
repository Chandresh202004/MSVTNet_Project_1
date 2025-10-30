#!/usr/bin/env python
# MSVTNet Training Summary Generator for BCIC IV 2b
# Author: Chandresh202004
# Date: 2025-07-13

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import argparse
from datetime import datetime
import re
import sys
import traceback

print("Script started...")
print(f"Python version: {sys.version}")
print(f"Arguments received: {sys.argv}")

def load_subject_data(results_dir, subject_id, mode="session_dependent_2b"):
    """Load training results for a specific subject"""
    base_path = os.path.join(results_dir, mode)
    history_path = os.path.join(base_path, f"subject_{subject_id}_history.json")
    args_path = os.path.join(base_path, f"subject_{subject_id}_args.json")
    report_path = os.path.join(base_path, f"subject_{subject_id}_classification_report.txt")
    
    print(f"Checking files for Subject {subject_id} in {mode}:")
    print(f"  - History file exists: {os.path.exists(history_path)}")
    print(f"  - Args file exists: {os.path.exists(args_path)}")
    print(f"  - Report file exists: {os.path.exists(report_path)}")
    
    data = {}
    
    # Load training history
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                data["history"] = json.load(f)
                print(f"  - Successfully loaded history data")
        except Exception as e:
            print(f"  - Error loading history file: {e}")
    else:
        print(f"Warning: History file not found for subject {subject_id} in {mode} mode")
    
    # Load model arguments
    if os.path.exists(args_path):
        try:
            with open(args_path, 'r') as f:
                data["args"] = json.load(f)
                print(f"  - Successfully loaded args data")
        except Exception as e:
            print(f"  - Error loading args file: {e}")
    else:
        print(f"Warning: Args file not found for subject {subject_id} in {mode} mode")
    
    # Load classification report
    if os.path.exists(report_path):
        try:
            with open(report_path, 'r') as f:
                data["report"] = f.read()
                print(f"  - Successfully loaded report data")
        except Exception as e:
            print(f"  - Error loading report file: {e}")
    else:
        print(f"Warning: Classification report not found for subject {subject_id} in {mode} mode")
    
    return data

def extract_metrics_from_report(report_text):
    """Extract key metrics from classification report text"""
    if not report_text:
        return {}
    
    lines = report_text.strip().split('\n')
    metrics = {}
    
    # Process class metrics (looking for 2 classes for BCIC IV 2b)
    class_names = ["Left Hand", "Right Hand"]
    for i, name in enumerate(class_names):
        for line in lines:
            if line.strip().startswith(name):
                parts = [p for p in line.split(' ') if p]
                try:
                    metrics[f"class_{i}_precision"] = float(parts[-3])
                    metrics[f"class_{i}_recall"] = float(parts[-2])
                    metrics[f"class_{i}_f1"] = float(parts[-1])
                except (IndexError, ValueError) as e:
                    print(f"Warning: Could not parse metrics for {name}: {e}")
                    print(f"Line content: '{line}'")
    
    # Extract accuracy
    for line in lines:
        if "accuracy" in line:
            try:
                # Use regex to find the accuracy value
                match = re.search(r'accuracy\s+(\d+\.\d+)', line)
                if match:
                    metrics["accuracy"] = float(match.group(1))
                    break
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse accuracy from line: '{line}', Error: {e}")
    
    return metrics

def generate_summary_table(subjects_data, mode):
    """Generate a summary table for all subjects"""
    headers = ["Subject", "Best Acc", "Final Train Acc", "Final Val Acc", 
               "Train Time", "Epochs", "LR", "Batch Size", "Dropout"]
    
    print(f"Generating summary table for {mode} with {len(subjects_data)} subjects")
    
    rows = []
    for subject_id, data in subjects_data.items():
        if not data or "history" not in data:
            print(f"No history data for subject {subject_id}")
            rows.append([subject_id, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
            continue
            
        history = data["history"]
        args = data.get("args", {})
        
        row = [
            subject_id,
            f"{history.get('best_acc', 0):.4f}",
            f"{history.get('train_accs', [0])[-1]:.4f}",
            f"{history.get('val_accs', [0])[-1]:.4f}",
            history.get('training_time', 'N/A'),
            args.get('epochs', 'N/A'),
            args.get('lr', 'N/A'),
            args.get('batch_size', 'N/A'),
            args.get('dropout', 'N/A')
        ]
        rows.append(row)
    
    # Calculate average metrics for the last row
    if rows:
        avg_row = ["Average"]
        for col in range(1, 4):  # Best Acc, Final Train Acc, Final Val Acc
            values = [float(row[col]) for row in rows if row[col] != "N/A"]
            avg = sum(values) / len(values) if values else 0
            avg_row.append(f"{avg:.4f}")
        avg_row.extend(["", "", "", "", ""])
        rows.append(avg_row)
    
    return tabulate(rows, headers=headers, tablefmt="grid")

def generate_class_performance_table(subjects_data):
    """Generate a table showing performance for each class across subjects"""
    headers = ["Subject", "Left Hand", "Right Hand", "Average"]  # Modified for BCIC IV 2b (2 classes)
    
    print(f"Generating class performance table with {len(subjects_data)} subjects")
    
    rows = []
    for subject_id, data in subjects_data.items():
        if not data or "report" not in data:
            print(f"No report data for subject {subject_id}")
            continue
            
        metrics = extract_metrics_from_report(data["report"])
        if not metrics or "accuracy" not in metrics:
            print(f"No valid metrics found in report for subject {subject_id}")
            continue
        
        recalls = []
        for i in range(2):  # 2 classes for BCIC IV 2b
            recalls.append(metrics.get(f"class_{i}_recall", 0))
            
        row = [
            subject_id,
            f"{recalls[0]:.2f}",
            f"{recalls[1]:.2f}",
            f"{metrics.get('accuracy', 0):.2f}"
        ]
        rows.append(row)
    
    # Calculate average for each class
    if rows:
        avg_row = ["Average"]
        for col in range(1, 4):  # 2 classes + average accuracy
            values = [float(row[col]) for row in rows]
            avg = sum(values) / len(values) if values else 0
            avg_row.append(f"{avg:.2f}")
        rows.append(avg_row)
    
    return tabulate(rows, headers=headers, tablefmt="grid")

def plot_accuracy_comparison(subjects_data, mode, output_dir):
    """Plot accuracy comparison between subjects"""
    plt.figure(figsize=(12, 6))
    
    best_accs = []
    subject_ids = []
    
    print(f"Plotting accuracy comparison for {mode}")
    
    for subject_id, data in sorted(subjects_data.items()):
        if not data or "history" not in data:
            print(f"No history data for subject {subject_id}")
            continue
            
        history = data["history"]
        best_acc = history.get('best_acc', 0)
        best_accs.append(best_acc)
        subject_ids.append(f"S{subject_id}")
    
    if not best_accs:
        print(f"Warning: No accuracy data found for {mode} mode")
        plt.close()
        return
        
    plt.bar(subject_ids, best_accs, color='blue', alpha=0.7)
    avg_acc = sum(best_accs)/len(best_accs)
    plt.axhline(y=avg_acc, color='r', linestyle='--', label=f'Average: {avg_acc:.3f}')
    plt.axhline(y=0.5, color='green', linestyle='--', label='Random Chance (0.5)')  # 0.5 for binary classification
    
    plt.ylim(0, max(1.0, max(best_accs) + 0.1))
    plt.xlabel('Subject')
    plt.ylabel('Best Validation Accuracy')
    plt.title(f'Subject Performance Comparison ({mode})')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f"{mode}_accuracy_comparison.png"))
    print(f"Saved subject accuracy comparison plot to {os.path.join(output_dir, f'{mode}_accuracy_comparison.png')}")
    plt.close()

def plot_training_curves(subjects_data, mode, output_dir, metric="accuracy"):
    """Plot training curves for all subjects"""
    plt.figure(figsize=(15, 10))
    
    # Define ylabel upfront to prevent reference before assignment
    ylabel = "Validation Accuracy" if metric == "accuracy" else "Validation Loss"
    
    print(f"Plotting training curves for {mode}, metric={metric}")
    
    has_data = False
    for subject_id, data in sorted(subjects_data.items()):
        if not data or "history" not in data:
            print(f"No history data for subject {subject_id}")
            continue
            
        history = data["history"]
        
        if metric == "accuracy":
            val_data = history.get('val_accs', [])
            label = f"Subject {subject_id} (Best: {history.get('best_acc', 0):.3f})"
        else:  # loss
            val_data = history.get('val_losses', [])
            label = f"Subject {subject_id}"
            
        if not val_data:
            print(f"No validation {metric} data for subject {subject_id}")
            continue
            
        x = list(range(1, len(val_data) + 1))
        plt.plot(x, val_data, label=label, alpha=0.8)
        has_data = True
    
    # Only proceed with labeling and saving if we have data
    if not has_data:
        print(f"Warning: No data available for {metric} training curves in {mode} mode")
        plt.close()
        return
    
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(f'Training Curves for All Subjects ({mode})')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f"{mode}_training_curves_{metric}.png"))
    print(f"Saved {metric} training curves plot to {os.path.join(output_dir, f'{mode}_training_curves_{metric}.png')}")
    plt.close()

def plot_combined_accuracy_comparison(subjects_data_dep, subjects_data_indep, output_dir):
    """Plot accuracy comparison between session-dependent and session-independent approaches"""
    plt.figure(figsize=(14, 8))
    
    # Prepare data
    subjects = range(1, 10)
    dep_accs = []
    indep_accs = []
    
    print("Plotting combined accuracy comparison")
    
    for subject_id in subjects:
        # Get session-dependent accuracy
        if subject_id in subjects_data_dep and subjects_data_dep[subject_id] and "history" in subjects_data_dep[subject_id]:
            dep_accs.append(subjects_data_dep[subject_id]["history"].get('best_acc', 0))
        else:
            print(f"No history data for subject {subject_id} in session-dependent mode")
            dep_accs.append(0)
            
        # Get session-independent accuracy
        if subject_id in subjects_data_indep and subjects_data_indep[subject_id] and "history" in subjects_data_indep[subject_id]:
            indep_accs.append(subjects_data_indep[subject_id]["history"].get('best_acc', 0))
        else:
            print(f"No history data for subject {subject_id} in session-independent mode")
            indep_accs.append(0)
    
    # Set up bar positions
    x = np.arange(len(subjects))
    width = 0.35
    
    # Create bars
    plt.bar(x - width/2, dep_accs, width, label='Session-Dependent', color='blue', alpha=0.7)
    plt.bar(x + width/2, indep_accs, width, label='Session-Independent', color='orange', alpha=0.7)
    
    # Calculate and plot averages
    avg_dep = sum(dep_accs) / len(dep_accs) if dep_accs else 0
    avg_indep = sum(indep_accs) / len(indep_accs) if indep_accs else 0
    
    plt.axhline(y=avg_dep, color='blue', linestyle='--', alpha=0.8, label=f'Avg Dep: {avg_dep:.3f}')
    plt.axhline(y=avg_indep, color='orange', linestyle='--', alpha=0.8, label=f'Avg Indep: {avg_indep:.3f}')
    plt.axhline(y=0.5, color='green', linestyle='--', label='Random Chance (0.5)')  # 0.5 for binary classification
    
    # Customize plot
    plt.ylim(0, max(1.0, max(max(dep_accs), max(indep_accs)) + 0.1))
    plt.xlabel('Subject')
    plt.ylabel('Best Validation Accuracy')
    plt.title('Performance Comparison: Session-Dependent vs Session-Independent')
    plt.xticks(x, [f"S{i}" for i in subjects])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for i, v in enumerate(dep_accs):
        plt.text(i - width/2, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=9, rotation=0)
        
    for i, v in enumerate(indep_accs):
        plt.text(i + width/2, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=9, rotation=0)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "combined_accuracy_comparison.png"))
    print(f"Saved combined accuracy comparison plot to {os.path.join(output_dir, 'combined_accuracy_comparison.png')}")
    plt.close()

def plot_class_performance(subjects_data, mode, output_dir):
    """Plot class-wise performance across subjects"""
    # Extract class performance for each subject
    class_data = {
        "Left Hand": [],
        "Right Hand": []
    }
    subject_ids = []
    
    print(f"Plotting class performance for {mode}")
    
    for subject_id, data in sorted(subjects_data.items()):
        if not data or "report" not in data:
            print(f"No report data for subject {subject_id}")
            continue
            
        metrics = extract_metrics_from_report(data["report"])
        if not metrics:
            print(f"No metrics extracted for subject {subject_id}")
            continue
            
        subject_ids.append(f"S{subject_id}")
        class_data["Left Hand"].append(metrics.get("class_0_recall", 0))
        class_data["Right Hand"].append(metrics.get("class_1_recall", 0))
    
    if not subject_ids:
        print(f"Warning: No class performance data found for {mode} mode")
        return
    
    # Plot class performance
    plt.figure(figsize=(14, 8))
    bar_width = 0.35
    x = np.arange(len(subject_ids))
    
    plt.bar(x - bar_width/2, class_data["Left Hand"], bar_width, label='Left Hand', color='blue', alpha=0.7)
    plt.bar(x + bar_width/2, class_data["Right Hand"], bar_width, label='Right Hand', color='green', alpha=0.7)
    
    plt.xlabel('Subject')
    plt.ylabel('Recall')
    plt.title(f'Class-wise Recall Performance ({mode})')
    plt.xticks(x, subject_ids)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 1.0)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f"{mode}_class_performance.png"))
    print(f"Saved class performance plot to {os.path.join(output_dir, f'{mode}_class_performance.png')}")
    plt.close()

def generate_summary_report(results_dir, output_dir, modes=None):
    """Generate comprehensive summary report"""
    if modes is None:
        modes = ["session_dependent_2b", "session_independent_2b"]
    
    print(f"\nGenerating MSVTNet training summary for BCIC IV 2b {', '.join(modes)} modes...\n")
    
    # Check directories
    print(f"Results directory exists: {os.path.exists(results_dir)}")
    if os.path.exists(results_dir):
        print(f"Files in results directory: {os.listdir(results_dir)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Store data for combined comparison
    all_subjects_data = {}
    
    # Generate report for each mode
    for mode in modes:
        print(f"\nProcessing {mode} mode...")
        
        # Check mode directory
        mode_dir = os.path.join(results_dir, mode)
        print(f"Mode directory {mode_dir} exists: {os.path.exists(mode_dir)}")
        if os.path.exists(mode_dir):
            print(f"Files in mode directory: {os.listdir(mode_dir)}")
        
        subjects_data = {}
        
        # Load data for all subjects
        for subject_id in range(1, 10):
            print(f"  Loading data for Subject {subject_id}...")
            subjects_data[subject_id] = load_subject_data(results_dir, subject_id, mode)
        
        all_subjects_data[mode] = subjects_data
        
        # Generate tables
        print("  Generating performance tables...")
        summary_table = generate_summary_table(subjects_data, mode)
        class_table = generate_class_performance_table(subjects_data)
        
        # Create plots
        print("  Creating visualization plots...")
        try:
            plot_accuracy_comparison(subjects_data, mode, output_dir)
            plot_training_curves(subjects_data, mode, output_dir, "accuracy")
            plot_training_curves(subjects_data, mode, output_dir, "loss")
            plot_class_performance(subjects_data, mode, output_dir)
        except Exception as e:
            print(f"Error during plot creation: {e}")
            traceback.print_exc()
        
        # Write report to file
        report_file = os.path.join(output_dir, f"{mode}_summary_report.txt")
        print(f"  Writing summary report to {report_file}...")
        
        try:
            with open(report_file, 'w') as f:
                f.write(f"MSVTNet BCIC IV 2b Training Results Summary - {mode}\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"User: {os.environ.get('USERNAME', 'Chandresh202004')}\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("1. Performance Summary\n")
                f.write("=" * 80 + "\n\n")
                f.write(summary_table + "\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("2. Class-wise Recall Performance\n")
                f.write("=" * 80 + "\n\n")
                f.write(class_table + "\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("3. Training Analysis\n")
                f.write("=" * 80 + "\n\n")
                
                avg_acc = 0
                count = 0
                best_subject = {"id": None, "acc": 0}
                worst_subject = {"id": None, "acc": 1}
                
                for subject_id, data in subjects_data.items():
                    if not data or "history" not in data:
                        continue
                        
                    acc = data["history"].get("best_acc", 0)
                    avg_acc += acc
                    count += 1
                    
                    if acc > best_subject["acc"]:
                        best_subject = {"id": subject_id, "acc": acc}
                        
                    if acc < worst_subject["acc"]:
                        worst_subject = {"id": subject_id, "acc": acc}
                
                if count > 0:
                    avg_acc /= count
                    
                    f.write(f"- Average accuracy across subjects: {avg_acc:.4f}\n")
                    f.write(f"- Best performing subject: Subject {best_subject['id']} ({best_subject['acc']:.4f})\n")
                    f.write(f"- Worst performing subject: Subject {worst_subject['id']} ({worst_subject['acc']:.4f})\n")
                    f.write(f"- Performance relative to chance (0.5): {(avg_acc - 0.5) / 0.5 * 100:.2f}% improvement\n\n")
                    
                    # Analyze overfitting
                    avg_train_acc = 0
                    for subject_id, data in subjects_data.items():
                        if not data or "history" not in data:
                            continue
                        
                        train_acc = data["history"].get("train_accs", [0])[-1]
                        avg_train_acc += train_acc
                    
                    avg_train_acc /= count
                    overfitting_gap = avg_train_acc - avg_acc
                    
                    f.write(f"- Average final training accuracy: {avg_train_acc:.4f}\n")
                    f.write(f"- Training-validation gap (overfitting): {overfitting_gap:.4f}\n")
                    if overfitting_gap > 0.4:
                        f.write("  (Severe overfitting detected)\n")
                    elif overfitting_gap > 0.2:
                        f.write("  (Moderate overfitting detected)\n")
                    else:
                        f.write("  (Minimal overfitting detected)\n")
                
                f.write("\n")
                f.write("Refer to the generated plots for visual comparison of subject performances.\n")
                print(f"  Successfully wrote summary report to {report_file}")
        except Exception as e:
            print(f"Error writing report file: {e}")
            traceback.print_exc()
    
    # Generate combined comparison if both modes are available
    if "session_dependent_2b" in all_subjects_data and "session_independent_2b" in all_subjects_data:
        print("\nGenerating combined comparison between approaches...")
        try:
            plot_combined_accuracy_comparison(
                all_subjects_data["session_dependent_2b"],
                all_subjects_data["session_independent_2b"],
                output_dir
            )
            
            # Create combined report
            combined_report_file = os.path.join(output_dir, "combined_analysis_report.txt")
            print(f"Writing combined analysis report to {combined_report_file}...")
            
            with open(combined_report_file, 'w') as f:
                f.write("MSVTNet BCIC IV 2b Combined Analysis: Session-Dependent vs Session-Independent\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"User: {os.environ.get('USERNAME', 'Chandresh202004')}\n\n")
                
                # Calculate average accuracies
                dep_accs = []
                indep_accs = []
                
                for subject_id in range(1, 10):
                    dep_data = all_subjects_data["session_dependent_2b"].get(subject_id, {})
                    indep_data = all_subjects_data["session_independent_2b"].get(subject_id, {})
                    
                    if dep_data and "history" in dep_data:
                        dep_accs.append(dep_data["history"].get("best_acc", 0))
                    
                    if indep_data and "history" in indep_data:
                        indep_accs.append(indep_data["history"].get("best_acc", 0))
                
                avg_dep = sum(dep_accs) / len(dep_accs) if dep_accs else 0
                avg_indep = sum(indep_accs) / len(indep_accs) if indep_accs else 0
                
                f.write("=" * 80 + "\n")
                f.write("1. Overall Approach Comparison\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Session-Dependent Average Accuracy: {avg_dep:.4f}\n")
                f.write(f"Session-Independent Average Accuracy: {avg_indep:.4f}\n\n")
                
                diff = avg_indep - avg_dep
                if diff > 0:
                    f.write(f"Session-Independent approach performs better by {diff:.4f} ({diff/avg_dep*100:.2f}%)\n")
                else:
                    f.write(f"Session-Dependent approach performs better by {-diff:.4f} ({-diff/avg_indep*100:.2f}%)\n")
                
                f.write("\n")
                f.write("=" * 80 + "\n")
                f.write("2. Subject-by-Subject Comparison\n")
                f.write("=" * 80 + "\n\n")
                
                # Create a table comparing each subject
                f.write("| Subject | Session-Dependent | Session-Independent | Difference | Better Approach |\n")
                f.write("|---------|------------------|---------------------|------------|----------------|\n")
                
                for subject_id in range(1, 10):
                    dep_acc = 0
                    indep_acc = 0
                    
                    if subject_id in all_subjects_data["session_dependent_2b"] and "history" in all_subjects_data["session_dependent_2b"][subject_id]:
                        dep_acc = all_subjects_data["session_dependent_2b"][subject_id]["history"].get("best_acc", 0)
                    
                    if subject_id in all_subjects_data["session_independent_2b"] and "history" in all_subjects_data["session_independent_2b"][subject_id]:
                        indep_acc = all_subjects_data["session_independent_2b"][subject_id]["history"].get("best_acc", 0)
                    
                    diff = indep_acc - dep_acc
                    better = "Independent" if diff > 0 else "Dependent" if diff < 0 else "Tie"
                    
                    f.write(f"| Subject {subject_id} | {dep_acc:.4f} | {indep_acc:.4f} | {abs(diff):.4f} | {better} |\n")
                
                f.write("\n")
                f.write("=" * 80 + "\n")
                f.write("3. Recommendations\n")
                f.write("=" * 80 + "\n\n")
                
                # Provide recommendations based on analysis
                if avg_indep > avg_dep:
                    f.write("Based on the analysis:\n")
                    f.write("- Session-Independent approach generally performs better\n")
                    f.write("- Consider using session-independent training for deployment\n")
                else:
                    f.write("Based on the analysis:\n")
                    f.write("- Session-Dependent approach generally performs better\n")
                    f.write("- Consider using session-dependent training with calibration for deployment\n")
                
                f.write("\nRecommended next steps:\n")
                if abs(avg_indep - avg_dep) < 0.05:
                    f.write("- The performance difference between approaches is small. Consider using an ensemble of both approaches.\n")
                
                f.write("- For subjects with high variance between approaches, investigate subject-specific characteristics.\n")
                f.write("- Consider adding domain adaptation techniques to improve session transfer performance.\n")
                
                print(f"Successfully wrote combined analysis report to {combined_report_file}")
        except Exception as e:
            print(f"Error generating combined comparison: {e}")
            traceback.print_exc()
    
    print("\nSummary generation completed!")
    print(f"All reports and visualizations saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MSVTNet BCIC IV 2b training summary report")
    parser.add_argument("--results_dir", type=str, default="D:/MSVTNet_Project/results_2b/msvtnet_2b",
                        help="Directory containing results for different approaches")
    parser.add_argument("--output_dir", type=str, default="D:/MSVTNet_Project/summary_2b",
                        help="Directory to save summary report")
    parser.add_argument("--mode", type=str, choices=["session_dependent_2b", "session_independent_2b", "both"],
                        default="both", help="Which training mode to analyze")
    
    args = parser.parse_args()
    
    print("Before calling generate_summary_report...")
    modes = ["session_dependent_2b", "session_independent_2b"] if args.mode == "both" else [args.mode]
    
    try:
        generate_summary_report(args.results_dir, args.output_dir, modes)
        print("After generate_summary_report completed successfully")
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()