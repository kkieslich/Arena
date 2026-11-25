#! /usr/bin/env python3
import gc

from arena_runtime.env.episode_node import EpisodeNode

#: allocations (net of frees) between two young-generation collections, CPython's default is 700. The pedestrian roster is
#: rebuilt for every render (50 Hz and on every engine frame): several thousand messages that live one or two renders. At
#: 700 they outlive the young collections, reach the oldest generation and set off a full collection of the node's heap
#: every few renders; above one roster's size they are freed before a collection sees them.
GC_YOUNG_THRESHOLD = 20_000


def main(args: list[str] | None = None) -> None:
    del args
    gc.set_threshold(GC_YOUNG_THRESHOLD, *gc.get_threshold()[1:])
    EpisodeNode.run_main(aiomonitor=True)


if __name__ == '__main__':
    import time

    time.sleep(5)
    main()
