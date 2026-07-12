// Logging system for fastsim — mirrors pathsim LoggerManager
// Simple, no external dependencies. INFO/WARNING -> stdout, ERROR -> stderr.

use std::time::Instant;

/// Log levels matching Python logging module.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Debug = 10,
    Info = 20,
    Warning = 30,
    Error = 40,
    Disabled = 100,
}

/// Simple logger — singleton-like, stored per Simulation.
#[derive(Clone)]
pub struct Logger {
    pub enabled: bool,
    pub level: LogLevel,
    pub prefix: String,
}

impl Logger {
    pub fn new(enabled: bool, prefix: &str) -> Self {
        Self {
            enabled,
            level: if enabled { LogLevel::Info } else { LogLevel::Disabled },
            prefix: prefix.to_string(),
        }
    }

    pub fn disabled() -> Self {
        Self::new(false, "")
    }

    #[inline]
    fn timestamp() -> String {
        use std::time::SystemTime;
        let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default();
        let secs = now.as_secs() % 86400; // time of day
        format!("{:02}:{:02}:{:02}", secs / 3600, (secs % 3600) / 60, secs % 60)
    }

    #[inline]
    pub fn info(&self, msg: &str) {
        if self.enabled && self.level <= LogLevel::Info {
            // Routed through the single sink (issue #29). The default sink sends
            // INFO/WARNING to stdout (normal progress output, not errors) so a
            // host capturing stderr separately (pathview's Pyodide worker) does
            // not render them as errors.
            crate::utils::sink::emit(LogLevel::Info, &format!("{} - INFO - {}", Self::timestamp(), msg));
        }
    }

    #[inline]
    pub fn warning(&self, msg: &str) {
        if self.enabled && self.level <= LogLevel::Warning {
            crate::utils::sink::emit(LogLevel::Warning, &format!("{} - WARNING - {}", Self::timestamp(), msg));
        }
    }

    #[inline]
    pub fn error(&self, msg: &str) {
        // Errors go to the sink (default: stderr), always (even if logging
        // disabled) — a genuine engine error should never be silenced by the
        // per-Simulation `log=False` flag.
        if self.level <= LogLevel::Error {
            crate::utils::sink::emit(LogLevel::Error, &format!("{} - ERROR - {}", Self::timestamp(), msg));
        }
    }
}

// ======================================================================================
// Progress Tracker — mirrors pathsim ProgressTracker with ASCII bar + ETA
// ======================================================================================

/// Progress tracker with ASCII bar, ETA, and rate display.
/// Mirrors pathsim/utils/progresstracker.py.
pub struct ProgressTracker {
    pub description: String,
    pub total_duration: f64,
    pub stats: ProgressStats,
    logger: Logger,
    start_time: Instant,
    last_log_time: Instant,
    last_log_progress: f64,
    ema_rate: f64,
    min_log_interval: f64,
    update_log_every: f64,
    bar_width: usize,
    ema_alpha: f64,
    interrupted: bool,
}

/// Lightweight progress stats (used internally by ProgressTracker).
#[derive(Clone, Debug)]
pub struct ProgressStats {
    pub total_steps: usize,
    pub successful_steps: usize,
    pub rejected_steps: usize,
    pub runtime_ms: f64,
}

impl Default for ProgressStats {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressStats {
    pub fn new() -> Self {
        Self { total_steps: 0, successful_steps: 0, rejected_steps: 0, runtime_ms: 0.0 }
    }
}

impl ProgressTracker {
    pub fn new(total_duration: f64, description: &str, enabled: bool) -> Self {
        Self {
            description: description.to_string(),
            total_duration,
            stats: ProgressStats::new(),
            logger: Logger::new(enabled, "progress"),
            start_time: Instant::now(),
            last_log_time: Instant::now(),
            last_log_progress: 0.0,
            ema_rate: 0.0,
            min_log_interval: 1.0,
            update_log_every: 0.2,
            bar_width: 20,
            ema_alpha: 0.3,
            interrupted: false,
        }
    }

    pub fn start(&mut self) {
        self.start_time = Instant::now();
        self.last_log_time = self.start_time;
        self.logger.info(&format!(
            "STARTING -> {} (Duration: {:.2}s)", self.description, self.total_duration
        ));
    }

    pub fn update(&mut self, progress: f64, success: bool) {
        self.stats.total_steps += 1;
        if success {
            self.stats.successful_steps += 1;
        } else {
            self.stats.rejected_steps += 1;
        }

        // EMA rate
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            let instant_rate = self.stats.total_steps as f64 / elapsed;
            if self.ema_rate == 0.0 {
                self.ema_rate = instant_rate;
            } else {
                self.ema_rate = self.ema_alpha * instant_rate + (1.0 - self.ema_alpha) * self.ema_rate;
            }
        }

        // Check if we should log
        let now = Instant::now();
        let time_trigger = now.duration_since(self.last_log_time).as_secs_f64() >= self.min_log_interval;
        let progress_trigger = progress >= self.last_log_progress + self.update_log_every;

        if (time_trigger || progress_trigger) && progress > 0.0 {
            self.log_progress(progress, elapsed);
            self.last_log_time = now;
            self.last_log_progress = progress;
        }
    }

    pub fn interrupt(&mut self) {
        self.interrupted = true;
    }

    pub fn close(&mut self) {
        self.stats.runtime_ms = self.start_time.elapsed().as_secs_f64() * 1000.0;

        let status = if self.interrupted { "INTERRUPTED" } else { "FINISHED" };
        self.logger.info(&format!(
            "{} -> {} (total steps: {}, successful: {}, runtime: {:.1} ms)",
            status, self.description,
            self.stats.total_steps, self.stats.successful_steps,
            self.stats.runtime_ms
        ));
    }

    fn log_progress(&self, progress: f64, elapsed: f64) {
        let pct = (progress * 100.0) as usize;
        let filled = (progress * self.bar_width as f64) as usize;
        let empty = self.bar_width - filled;
        let bar: String = "#".repeat(filled) + &"-".repeat(empty);

        let elapsed_str = format_time(elapsed);
        let eta = if progress > 0.01 {
            format_time(elapsed / progress * (1.0 - progress))
        } else {
            "--:--".to_string()
        };
        let rate_str = format_rate(self.ema_rate);

        self.logger.info(&format!(
            "{} {:3}% | {}<{} | {}", bar, pct, elapsed_str, eta, rate_str
        ));
    }
}

fn format_time(secs: f64) -> String {
    if secs < 0.0 || secs.is_nan() || secs.is_infinite() {
        return "--:--".to_string();
    }
    if secs < 60.0 {
        format!("{:.1}s", secs)
    } else if secs < 3600.0 {
        format!("{:02}:{:02}", (secs / 60.0) as u64, (secs % 60.0) as u64)
    } else {
        format!("{:02}:{:02}:{:02}", (secs / 3600.0) as u64, ((secs % 3600.0) / 60.0) as u64, (secs % 60.0) as u64)
    }
}

fn format_rate(rate: f64) -> String {
    if rate <= 0.0 || rate.is_nan() {
        "N/A".to_string()
    } else if rate < 0.1 {
        format!("{:.1} it/min", rate * 60.0)
    } else if rate < 1.0 {
        format!("{:.2} it/s", rate)
    } else {
        format!("{:.1} it/s", rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_time() {
        assert_eq!(format_time(5.2), "5.2s");
        assert_eq!(format_time(65.0), "01:05");
        assert_eq!(format_time(3661.0), "01:01:01");
    }

    #[test]
    fn test_format_rate() {
        assert_eq!(format_rate(0.05), "3.0 it/min");
        assert_eq!(format_rate(0.5), "0.50 it/s");
        assert_eq!(format_rate(42.0), "42.0 it/s");
    }

    #[test]
    fn test_logger_levels() {
        let log = Logger::new(true, "test");
        assert!(log.enabled);
        assert!(log.level <= LogLevel::Info);

        let log = Logger::disabled();
        assert!(!log.enabled);
    }
}
