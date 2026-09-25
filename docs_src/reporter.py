import logging

from alpineagents import Reporter

log = logging.getLogger("agent")


class LogReporter(Reporter):
    def on_tool_start(self, state, call):
        log.info("turn %d: %s(%s)", state.turn, call.name, call.args)

    def on_run_end(self, state, error):
        log.info("stopped by %s after %d turns", state.stopped_by, state.turn)
