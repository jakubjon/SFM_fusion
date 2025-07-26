"""
Main orchestration script for the painting reconstruction pipeline.
This script coordinates all processing steps based on the configuration.
"""

import sys
from pathlib import Path
from datetime import datetime
import config

# Import all step modules
from step1_local_sfm import LocalSfMStep
from step2_global_calibration import GlobalCalibrationStep
from step3_recalculate_positions import RecalculatePositionsStep
from step4_point_cloud_generation import PointCloudGenerationStep
from step5_rectification import RectificationStep
from step6_manual_roi_selection import ManualROISelectionStep
from step7_high_res_rectification import HighResRectificationStep


class PaintingReconstructionPipeline:
    """Main pipeline orchestrator for painting reconstruction"""
    
    def __init__(self, photos_dir=config.PHOTOS_DIR, output_dir=config.OUTPUT_DIR):
        self.photos_dir = Path(photos_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize all steps
        self.steps = {
            'run_local_sfm': LocalSfMStep(photos_dir, output_dir),
            'global_calibration': GlobalCalibrationStep(output_dir),
            'recalculate_positions': RecalculatePositionsStep(photos_dir, output_dir),
            'point_cloud_generation': PointCloudGenerationStep(output_dir),
            'rectification': RectificationStep(photos_dir, output_dir),
            'manual_roi_selection': ManualROISelectionStep(photos_dir, output_dir),
            'high_res_rectification': HighResRectificationStep(photos_dir, output_dir)
        }
        
        # Step dependencies
        self.step_dependencies = {
            'run_local_sfm': [],
            'global_calibration': ['run_local_sfm'],
            'recalculate_positions': ['run_local_sfm', 'global_calibration'],
            'point_cloud_generation': ['recalculate_positions'],
            'rectification': ['recalculate_positions', 'point_cloud_generation'],
            'manual_roi_selection': ['rectification'],
            'high_res_rectification': ['recalculate_positions', 'point_cloud_generation', 'manual_roi_selection']
        }
    
    def log_pipeline_start(self):
        """Log pipeline start information"""
        print("=" * 80)
        print("PAINTING RECONSTRUCTION PIPELINE")
        print("=" * 80)
        print(f"Photos directory: {self.photos_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show enabled steps
        enabled_steps = [step for step, enabled in config.PROCESSING_STEPS.items() if enabled]
        print(f"\nEnabled steps: {enabled_steps}")
        
        if not enabled_steps:
            print("[ERROR] No steps are enabled. Please check config.PROCESSING_STEPS")
            return False
        
        return True
    
    def check_prerequisites(self):
        """Check if all prerequisites are met"""
        print("\nChecking prerequisites...")
        
        # Check if photos directory exists
        if not self.photos_dir.exists():
            print(f"[ERROR] Photos directory not found: {self.photos_dir}")
            return False
        
        # Check if there are any painting subdirectories
        painting_sets = [d for d in self.photos_dir.iterdir() if d.is_dir()]
        if not painting_sets:
            print(f"[ERROR] No painting subdirectories found in {self.photos_dir}")
            return False
        
        print(f"[OK] Found {len(painting_sets)} painting sets: {[p.name for p in painting_sets]}")
        return True
    
    def get_step_execution_order(self):
        """Determine the order in which steps should be executed based ONLY on PROCESSING_STEPS configuration"""
        enabled_steps = [step for step, enabled in config.PROCESSING_STEPS.items() if enabled]
        
        # Simple execution order - only run explicitly enabled steps
        # Dependencies are the responsibility of the user to configure correctly
        execution_order = []
        visited = set()
        temp_visited = set()
        
        def visit(step):
            if step in temp_visited:
                raise ValueError(f"Circular dependency detected involving step: {step}")
            if step in visited:
                return
            
            temp_visited.add(step)
            
            # Only visit dependencies that are also explicitly enabled
            for dep in self.step_dependencies.get(step, []):
                if dep in enabled_steps and dep not in visited:
                    visit(dep)
            
            temp_visited.remove(step)
            visited.add(step)
            execution_order.append(step)
        
        for step in enabled_steps:
            if step not in visited:
                visit(step)
        
        return execution_order
    
    def run_step(self, step_name):
        """Run a single step"""
        print(f"\n{'='*80}")
        print(f"RUNNING STEP: {step_name}")
        print(f"{'='*80}")
        
        if step_name not in self.steps:
            print(f"[ERROR] Unknown step: {step_name}")
            return False
        
        step = self.steps[step_name]
        
        try:
            # Run the step
            result = step.run()
            
            if result is None:
                print(f"[ERROR] Step {step_name} failed")
                return False
            
            print(f"[OK] Step {step_name} completed successfully")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error in step {step_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_pipeline(self):
        """Run the complete pipeline"""
        if not self.log_pipeline_start():
            return False
        
        if not self.check_prerequisites():
            return False
        
        # Determine execution order
        execution_order = self.get_step_execution_order()
        print(f"\nExecution order: {execution_order}")
        
        # Run steps in order
        successful_steps = []
        failed_steps = []
        
        for step_name in execution_order:
            if self.run_step(step_name):
                successful_steps.append(step_name)
            else:
                failed_steps.append(step_name)
                print(f"\n[ERROR] Pipeline failed at step: {step_name}")
                break
        
        # Print summary
        print(f"\n{'='*80}")
        print("PIPELINE SUMMARY")
        print(f"{'='*80}")
        print(f"Successful steps: {successful_steps}")
        if failed_steps:
            print(f"Failed steps: {failed_steps}")
            print(f"[ERROR] Pipeline failed after {len(successful_steps)} steps")
            return False
        else:
            print(f"[OK] Pipeline completed successfully!")
        print(f"Results saved in: {self.output_dir}")
        return True
    
    def run_single_step(self, step_name):
        """Run a single step independently"""
        if step_name not in config.PROCESSING_STEPS:
            print(f"[ERROR] Unknown step: {step_name}")
            return False
        
        if not config.PROCESSING_STEPS[step_name]:
            print(f"[ERROR] Step {step_name} is disabled in configuration")
            return False
        
        print(f"Running single step: {step_name}")
        return self.run_step(step_name)
    
    def list_steps(self):
        """List all available steps and their status"""
        print("Available steps:")
        for step_name, enabled in config.PROCESSING_STEPS.items():
            status = "[OK] ENABLED" if enabled else "[ERROR] DISABLED"
            print(f"  {step_name}: {status}")
    
    def check_step_dependencies(self, step_name):
        """Check dependencies for a specific step"""
        if step_name not in self.step_dependencies:
            print(f"[ERROR] Unknown step: {step_name}")
            return
        
        dependencies = self.step_dependencies[step_name]
        print(f"Dependencies for step '{step_name}':")
        if dependencies:
            for dep in dependencies:
                status = "[OK] ENABLED" if config.PROCESSING_STEPS.get(dep, False) else "[ERROR] DISABLED"
                print(f"  {dep}: {status}")
        else:
            print("  No dependencies")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Painting Reconstruction Pipeline")
    parser.add_argument("--step", help="Run a single step")
    parser.add_argument("--list-steps", action="store_true", help="List all available steps")
    parser.add_argument("--check-deps", help="Check dependencies for a specific step")
    parser.add_argument("--photos-dir", default=config.PHOTOS_DIR, help="Photos directory")
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR, help="Output directory")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = PaintingReconstructionPipeline(args.photos_dir, args.output_dir)
    
    if args.list_steps:
        pipeline.list_steps()
        return
    
    if args.check_deps:
        pipeline.check_step_dependencies(args.check_deps)
        return
    
    if args.step:
        # Run single step
        success = pipeline.run_single_step(args.step)
        sys.exit(0 if success else 1)
    else:
        # Run complete pipeline
        success = pipeline.run_pipeline()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
