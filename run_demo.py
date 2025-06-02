#!/usr/bin/env python
# coding: utf-8

"""
Demo script for CS365 Arabic Text Classification Project
Run this from the root directory to test all models
"""

from src.utils.comparison import quick_inference_demo, compare_models

def main():
    print("CS365 Arabic Text Classification Demo")
    print("=" * 50)
    
    print("\n1. Quick Inference Demo:")
    quick_inference_demo()
    
    print("\n" + "=" * 50)
    print("\nTo run full model comparison, uncomment the line below:")
    print("# compare_models()")
    
    # Uncomment to run full comparison:
    compare_models()

if __name__ == "__main__":
    main() 