# Stock Dashboard Configuration
from stockDashboard.layout import layout as stockDashboardLayout
from stockDashboard.callbacks import register_callbacks as stockDashboardCallbacks

# CNN Dashboard Configuration
from cnnDashboard.layout import layout as cnnDashboardLayout
from cnnDashboard.callbacks import register_callbacks as cnnDashboardCallbacks

appdict = {
    "dashboard": {
        "title": "Dashboard",
        "layout": stockDashboardLayout,
        "callbacks": stockDashboardCallbacks,
    },
    "cnn": {
        "title": "CNN",
        "layout": cnnDashboardLayout,
        "callbacks": cnnDashboardCallbacks,
    }
}
