"""Advanced alerting system for MLOps monitoring."""
import logging
import json
import smtplib
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yaml
import redis
from sqlalchemy import create_engine, text
from dataclasses import dataclass, asdict
from jinja2 import Template


logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Base alert class."""
    alert_id: str
    alert_type: str
    severity: str
    title: str
    description: str
    model_name: str
    environment: str
    timestamp: datetime
    resolved: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass  
class NotificationChannel:
    """Notification channel configuration."""
    name: str
    type: str  # email, slack, webhook, pagerduty
    config: Dict[str, Any]
    enabled: bool = True
    severity_filter: List[str] = None  # Which severities to notify
    
    def __post_init__(self):
        if self.severity_filter is None:
            self.severity_filter = ["low", "medium", "high", "critical"]


class AlertManager:
    """Comprehensive alerting system for MLOps monitoring."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize alert manager."""
        self.config = self._load_config(config_path)
        self.db_engine = create_engine(self.config["database"]["connection_string"])
        self.redis_client = redis.from_url(self.config["redis"]["connection_string"])
        
        # Load notification channels
        self.notification_channels = self._load_notification_channels()
        
        # Alert templates
        self.templates = self._load_alert_templates()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
            
    def _load_notification_channels(self) -> List[NotificationChannel]:
        """Load notification channel configurations."""
        channels = []
        
        # Email configuration
        if self.config.get("monitoring", {}).get("alerts", {}).get("email_notifications"):
            email_config = {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "alerts@company.com",
                "password": "app_password",  # Use app password or OAuth
                "recipients": ["admin@company.com", "mlops-team@company.com"]
            }
            channels.append(NotificationChannel(
                name="email",
                type="email",
                config=email_config,
                severity_filter=["medium", "high", "critical"]
            ))
            
        # Slack configuration
        if self.config.get("monitoring", {}).get("alerts", {}).get("slack_notifications"):
            slack_config = {
                "webhook_url": self.config["monitoring"]["alerts"].get("webhook_url", ""),
                "channel": "#mlops-alerts",
                "username": "MLOps Alert Bot"
            }
            channels.append(NotificationChannel(
                name="slack",
                type="slack",
                config=slack_config,
                severity_filter=["high", "critical"]
            ))
            
        # PagerDuty for critical alerts
        pagerduty_config = {
            "integration_key": "YOUR_PAGERDUTY_INTEGRATION_KEY",
            "service_key": "YOUR_PAGERDUTY_SERVICE_KEY"
        }
        channels.append(NotificationChannel(
            name="pagerduty",
            type="pagerduty", 
            config=pagerduty_config,
            severity_filter=["critical"]
        ))
        
        return channels
        
    def _load_alert_templates(self) -> Dict[str, Template]:
        """Load alert message templates."""
        templates = {}
        
        # Email templates
        templates["email_drift_alert"] = Template("""
        <h2>🚨 Data Drift Alert</h2>
        <p><strong>Model:</strong> {{ model_name }}</p>
        <p><strong>Environment:</strong> {{ environment }}</p>
        <p><strong>Severity:</strong> {{ severity }}</p>
        <p><strong>Features with drift:</strong> {{ affected_features|join(', ') }}</p>
        
        <h3>Details:</h3>
        <ul>
        {% for feature, details in feature_details.items() %}
            <li><strong>{{ feature }}</strong>: Drift score {{ "%.4f"|format(details.drift_score) }} (p-value: {{ "%.4f"|format(details.p_value) }})</li>
        {% endfor %}
        </ul>
        
        <p><strong>Timestamp:</strong> {{ timestamp }}</p>
        <p>Please investigate the data quality and consider retraining the model.</p>
        """)
        
        templates["email_performance_alert"] = Template("""
        <h2>⚠️ Model Performance Alert</h2>
        <p><strong>Model:</strong> {{ model_name }} ({{ model_version }})</p>
        <p><strong>Environment:</strong> {{ environment }}</p>
        <p><strong>Severity:</strong> {{ severity }}</p>
        <p><strong>Metric:</strong> {{ metric_name }}</p>
        
        <h3>Performance Details:</h3>
        <ul>
            <li><strong>Current Value:</strong> {{ "%.4f"|format(current_value) }}</li>
            <li><strong>Baseline Value:</strong> {{ "%.4f"|format(baseline_value) }}</li>
            <li><strong>Degradation:</strong> {{ "%.4f"|format(details.degradation) }} ({{ "%.1f"|format(details.degradation_percent) }}%)</li>
            <li><strong>Threshold:</strong> {{ "%.4f"|format(threshold) }}</li>
        </ul>
        
        <p><strong>Timestamp:</strong> {{ timestamp }}</p>
        <p>Model performance has degraded below acceptable thresholds. Consider investigating and potentially rolling back or retraining.</p>
        """)
        
        # Slack templates
        templates["slack_drift_alert"] = Template("""
        :warning: *Data Drift Alert*
        
        *Model:* {{ model_name }}
        *Environment:* {{ environment }}
        *Severity:* {{ severity }}
        *Features:* {{ affected_features|join(', ') }}
        
        *Top Drifted Features:*
        {% for feature, details in feature_details.items() %}
        • {{ feature }}: {{ "%.4f"|format(details.drift_score) }}
        {% endfor %}
        
        _{{ timestamp }}_
        """)
        
        templates["slack_performance_alert"] = Template("""
        :rotating_light: *Model Performance Alert*
        
        *Model:* {{ model_name }} ({{ model_version }})
        *Environment:* {{ environment }}
        *Severity:* {{ severity }}
        *Metric:* {{ metric_name }}
        
        *Performance:*
        • Current: {{ "%.4f"|format(current_value) }}
        • Baseline: {{ "%.4f"|format(baseline_value) }}
        • Degradation: {{ "%.1f"|format(details.degradation_percent) }}%
        
        _{{ timestamp }}_
        """)
        
        return templates
        
    def create_alert(self, alert: Alert) -> str:
        """Create and store a new alert."""
        try:
            # Generate alert ID if not provided
            if not alert.alert_id:
                alert.alert_id = f"{alert.alert_type}_{alert.model_name}_{int(alert.timestamp.timestamp())}"
                
            # Store alert in database
            with self.db_engine.connect() as conn:
                query = text("""
                    INSERT INTO monitoring.alerts 
                    (alert_id, alert_type, severity, title, description, model_name, 
                     environment, timestamp, resolved, metadata)
                    VALUES 
                    (:alert_id, :alert_type, :severity, :title, :description, :model_name,
                     :environment, :timestamp, :resolved, :metadata)
                """)
                
                # Create table if it doesn't exist
                create_table_query = text("""
                    CREATE TABLE IF NOT EXISTS monitoring.alerts (
                        id SERIAL PRIMARY KEY,
                        alert_id VARCHAR(255) UNIQUE NOT NULL,
                        alert_type VARCHAR(100) NOT NULL,
                        severity VARCHAR(20) NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT,
                        model_name VARCHAR(255) NOT NULL,
                        environment VARCHAR(50) NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        resolved BOOLEAN DEFAULT FALSE,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                try:
                    conn.execute(create_table_query)
                except Exception:
                    pass  # Table might already exist
                    
                conn.execute(query, {
                    "alert_id": alert.alert_id,
                    "alert_type": alert.alert_type,
                    "severity": alert.severity,
                    "title": alert.title,
                    "description": alert.description,
                    "model_name": alert.model_name,
                    "environment": alert.environment,
                    "timestamp": alert.timestamp,
                    "resolved": alert.resolved,
                    "metadata": json.dumps(alert.metadata)
                })
                conn.commit()
                
            # Queue alert for notification
            self._queue_alert_for_notification(alert)
            
            logger.info(f"Created alert: {alert.alert_id}")
            return alert.alert_id
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            raise
            
    def _queue_alert_for_notification(self, alert: Alert) -> None:
        """Queue alert for notification processing."""
        try:
            alert_data = asdict(alert)
            alert_data["timestamp"] = alert.timestamp.isoformat()
            
            self.redis_client.lpush("alert_notifications", json.dumps(alert_data, default=str))
            logger.info(f"Queued alert for notification: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to queue alert for notification: {e}")
            
    def process_notifications(self) -> None:
        """Process queued alert notifications."""
        try:
            while True:
                # Get next alert from queue (blocking with timeout)
                result = self.redis_client.brpop("alert_notifications", timeout=5)
                
                if not result:
                    break
                    
                _, alert_data_json = result
                alert_data = json.loads(alert_data_json.decode())
                
                # Convert back to Alert object
                alert_data["timestamp"] = datetime.fromisoformat(alert_data["timestamp"])
                alert = Alert(**alert_data)
                
                # Send notifications
                self._send_notifications(alert)
                
        except Exception as e:
            logger.error(f"Error processing notifications: {e}")
            
    def _send_notifications(self, alert: Alert) -> None:
        """Send notifications through configured channels."""
        for channel in self.notification_channels:
            if not channel.enabled:
                continue
                
            if alert.severity not in channel.severity_filter:
                continue
                
            try:
                if channel.type == "email":
                    self._send_email_notification(alert, channel)
                elif channel.type == "slack":
                    self._send_slack_notification(alert, channel)
                elif channel.type == "pagerduty":
                    self._send_pagerduty_notification(alert, channel)
                elif channel.type == "webhook":
                    self._send_webhook_notification(alert, channel)
                    
            except Exception as e:
                logger.error(f"Failed to send notification via {channel.name}: {e}")
                
    def _send_email_notification(self, alert: Alert, channel: NotificationChannel) -> None:
        """Send email notification."""
        try:
            config = channel.config
            
            # Create message
            msg = MIMEMultipart()
            msg["From"] = config["username"]
            msg["To"] = ", ".join(config["recipients"])
            msg["Subject"] = f"[{alert.severity.upper()}] {alert.title}"
            
            # Generate email body from template
            template_key = f"email_{alert.alert_type}"
            if template_key in self.templates:
                template = self.templates[template_key]
                body = template.render(**alert.metadata, **asdict(alert))
            else:
                body = f"""
                Alert: {alert.title}
                
                Model: {alert.model_name}
                Environment: {alert.environment}
                Severity: {alert.severity}
                Description: {alert.description}
                Timestamp: {alert.timestamp}
                
                Metadata: {json.dumps(alert.metadata, indent=2)}
                """
                
            msg.attach(MIMEText(body, "html"))
            
            # Send email
            server = smtplib.SMTP(config["smtp_server"], config["smtp_port"])
            server.starttls()
            server.login(config["username"], config["password"])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Sent email notification for alert: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            
    def _send_slack_notification(self, alert: Alert, channel: NotificationChannel) -> None:
        """Send Slack notification."""
        try:
            config = channel.config
            
            # Generate Slack message from template
            template_key = f"slack_{alert.alert_type}"
            if template_key in self.templates:
                template = self.templates[template_key]
                text = template.render(**alert.metadata, **asdict(alert))
            else:
                text = f"*{alert.title}*\n\nModel: {alert.model_name}\nEnvironment: {alert.environment}\nSeverity: {alert.severity}\n\n{alert.description}"
                
            # Slack payload
            payload = {
                "channel": config["channel"],
                "username": config["username"],
                "text": text,
                "icon_emoji": self._get_severity_emoji(alert.severity)
            }
            
            # Send to Slack
            response = requests.post(config["webhook_url"], json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Sent Slack notification for alert: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            
    def _send_pagerduty_notification(self, alert: Alert, channel: NotificationChannel) -> None:
        """Send PagerDuty notification."""
        try:
            config = channel.config
            
            payload = {
                "routing_key": config["integration_key"],
                "event_action": "trigger",
                "dedup_key": alert.alert_id,
                "payload": {
                    "summary": alert.title,
                    "source": f"{alert.model_name} ({alert.environment})",
                    "severity": alert.severity,
                    "component": alert.model_name,
                    "group": "mlops",
                    "class": alert.alert_type,
                    "custom_details": alert.metadata
                }
            }
            
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"Sent PagerDuty notification for alert: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send PagerDuty notification: {e}")
            
    def _send_webhook_notification(self, alert: Alert, channel: NotificationChannel) -> None:
        """Send webhook notification."""
        try:
            config = channel.config
            
            payload = {
                "alert": asdict(alert),
                "timestamp": alert.timestamp.isoformat()
            }
            
            response = requests.post(
                config["webhook_url"],
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"Sent webhook notification for alert: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            
    def _get_severity_emoji(self, severity: str) -> str:
        """Get emoji for severity level."""
        emoji_map = {
            "low": ":information_source:",
            "medium": ":warning:",
            "high": ":exclamation:",
            "critical": ":rotating_light:"
        }
        return emoji_map.get(severity, ":question:")
        
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an alert."""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    UPDATE monitoring.alerts 
                    SET resolved = TRUE, resolved_by = :resolved_by, resolved_at = :resolved_at
                    WHERE alert_id = :alert_id
                """)
                
                # Add resolved_by and resolved_at columns if they don't exist
                try:
                    alter_query = text("""
                        ALTER TABLE monitoring.alerts 
                        ADD COLUMN IF NOT EXISTS resolved_by VARCHAR(255),
                        ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMP
                    """)
                    conn.execute(alter_query)
                except Exception:
                    pass
                    
                result = conn.execute(query, {
                    "alert_id": alert_id,
                    "resolved_by": resolved_by,
                    "resolved_at": datetime.now()
                })
                conn.commit()
                
            if result.rowcount > 0:
                logger.info(f"Resolved alert: {alert_id}")
                return True
            else:
                logger.warning(f"Alert not found: {alert_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to resolve alert: {e}")
            return False
            
    def get_active_alerts(self, model_name: Optional[str] = None,
                         environment: Optional[str] = None,
                         severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active alerts with optional filtering."""
        try:
            query = "SELECT * FROM monitoring.alerts WHERE resolved = FALSE"
            params = {}
            
            if model_name:
                query += " AND model_name = :model_name"
                params["model_name"] = model_name
                
            if environment:
                query += " AND environment = :environment"
                params["environment"] = environment
                
            if severity:
                query += " AND severity = :severity"
                params["severity"] = severity
                
            query += " ORDER BY timestamp DESC"
            
            with self.db_engine.connect() as conn:
                results = conn.execute(text(query), params).fetchall()
                
            alerts = []
            for row in results:
                alert_dict = dict(row._mapping)
                # Parse metadata JSON
                if alert_dict.get("metadata"):
                    try:
                        alert_dict["metadata"] = json.loads(alert_dict["metadata"])
                    except json.JSONDecodeError:
                        alert_dict["metadata"] = {}
                alerts.append(alert_dict)
                
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to get active alerts: {e}")
            return []
            
    def get_alert_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get alert summary for dashboard."""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT 
                        severity,
                        alert_type,
                        COUNT(*) as count,
                        SUM(CASE WHEN resolved THEN 1 ELSE 0 END) as resolved_count
                    FROM monitoring.alerts 
                    WHERE timestamp >= :since_date
                    GROUP BY severity, alert_type
                    ORDER BY count DESC
                """)
                
                results = conn.execute(query, {
                    "since_date": datetime.now() - timedelta(days=days)
                }).fetchall()
                
            summary = {
                "period_days": days,
                "total_alerts": sum(row.count for row in results),
                "resolved_alerts": sum(row.resolved_count for row in results),
                "by_severity": {},
                "by_type": {}
            }
            
            for row in results:
                severity = row.severity
                alert_type = row.alert_type
                
                if severity not in summary["by_severity"]:
                    summary["by_severity"][severity] = {"total": 0, "resolved": 0}
                summary["by_severity"][severity]["total"] += row.count
                summary["by_severity"][severity]["resolved"] += row.resolved_count
                
                if alert_type not in summary["by_type"]:
                    summary["by_type"][alert_type] = {"total": 0, "resolved": 0}
                summary["by_type"][alert_type]["total"] += row.count
                summary["by_type"][alert_type]["resolved"] += row.resolved_count
                
            # Calculate resolution rate
            total_alerts = summary["total_alerts"]
            resolved_alerts = summary["resolved_alerts"]
            summary["resolution_rate"] = (resolved_alerts / total_alerts * 100) if total_alerts > 0 else 0
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get alert summary: {e}")
            return {"error": str(e)}