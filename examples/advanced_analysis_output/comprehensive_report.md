# OECT Transfer Analysis - Comprehensive Report

## Dataset Overview
- Total transfer curves: 100
- Analysis date: 2025-07-03 14:00:47
- Average data points per curve: 1

## Stability Analysis Summary
- Parameters analyzed: 15
- Stable parameters (drift < 3%): 0
- Unstable parameters: 15

## Key Findings
- gm_max_raw: increase by 2219.87%
- I_max_raw: increase by 571.31%
- Von_raw: decrease by 39.44%
- absgm_max_raw: increase by 2244.15%

## Parameter Stability Ranking (by CV%)
Most stable:
  1. Von_raw: -607.98%
  2. Von_forward: -59.15%
  3. Von_reverse: 72.56%
  4. I_max_forward: 87.51%
  5. absI_max_forward: 87.51%

Least stable:
  1. absgm_max_raw: 106.87%
  2. absgm_max_reverse: 106.87%
  3. gm_max_forward: 112.86%
  4. gm_max_raw: 113.50%
  5. gm_max_reverse: 113.50%

## Recommendations
- Monitor parameters with CV > 10%: 19 parameters
- Investigate trends in unstable parameters
- Consider environmental factors for drift > 5%