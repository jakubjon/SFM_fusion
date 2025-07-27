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
        
        # Determine execution steps
        enabled_steps = [step for step, enabled in config.PROCESSING_STEPS.items() if enabled]
        print(f"\nExecution order: {enabled_steps}")
        
        # Run steps in order
        successful_steps = []
        failed_steps = []
        
        for step_name in enabled_steps:
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
            print("[OK] Pipeline completed successfully!")
        print(f"Results saved in: {self.output_dir}")
        return True
      
    def list_steps(self):
        """List all available steps and their status"""
        print("Available steps:")
        for step_name, enabled in config.PROCESSING_STEPS.items():
            status = "[OK] ENABLED" if enabled else "[ERROR] DISABLED"
            print(f"  {step_name}: {status}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Painting Reconstruction Pipeline")
    parser.add_argument("--step", help="Run a single step")
    parser.add_argument("--list-steps", action="store_true", help="List all available steps")
    parser.add_argument("--photos-dir", default=config.PHOTOS_DIR, help="Photos directory")
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR, help="Output directory")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = PaintingReconstructionPipeline(args.photos_dir, args.output_dir)
    
    if args.list_steps:
        pipeline.list_steps()
        return
    
    if args.step:
        # Run single step
        success = pipeline.run_step(args.step)
        sys.exit(0 if success else 1)
    else:
        # Run complete pipeline
        success = pipeline.run_pipeline()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
