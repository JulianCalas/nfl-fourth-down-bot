"""NFL fourth-down decision bot.

The package builds four pieces from play-by-play, none of which are copied
from nflfastR's precomputed win probability or expected points:

* a win-probability model
* a fourth-down conversion model
* a field-goal make model
* a punt outcome distribution

Those pieces are combined into the expected win probability of going for it,
punting, and kicking. Coach decisions are then graded against that recommendation.
"""

__version__ = "1.0.0"
