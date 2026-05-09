import re

with open("templates/index.before_layout.html", "r", encoding="utf-8") as f:
    old_html = f.read()
    
script_start = old_html.find("<script>")
script_block = old_html[script_start:]

# Change DOMContentLoaded hook to load latest tasks by default
script_block = script_block.replace("loadUserPreferences();", "loadUserPreferences();\\n            loadLatestTasks();")

with open("templates/index.html", "r", encoding="utf-8") as f:
    new_html = f.read()

modal_start = new_html.find("<!-- Settings Modal -->")
if modal_start != -1:
    new_html = new_html[:modal_start]

new_html = new_html.replace("onclick=\"toggleSettings()\"", "href=\"/settings\"")
new_html = new_html.replace("<button id=\"settingsBtn\"", "<a id=\"settingsBtn\"")
new_html = new_html.replace("</button>\\n        </div>\\n    </header>", "</a>\\n        </div>\\n    </header>")

with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write(new_html + script_block)
