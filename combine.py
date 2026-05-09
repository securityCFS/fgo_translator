with open('templates/index.before_layout.html', 'r', encoding='utf-8') as f:
    orig = f.read()

script_start = orig.find('<script>')
script_block = orig[script_start:]
script_block = script_block.replace('loadUserPreferences();', 'loadUserPreferences();\n            loadLatestTasks();')

with open('templates/demo.html', 'r', encoding='utf-8') as f:
    demo = f.read()

# Replace demo's dummy scripts/modals
modal_start = demo.find('<!-- Settings Modal -->')
if modal_start != -1:
    demo = demo[:modal_start]

# Change settings button specifically
import re
demo = re.sub(
    r'<button id="settingsBtn" .*?</button>',
    r'<a id="settingsBtn" class="p-2 text-slate-500 hover:text-indigo-600 hover:bg-indigo-50 rounded-full transition-colors" href="/settings"><svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path></svg></a>',
    demo, flags=re.DOTALL
)

# Add custom JS functions needed for the UI
custom_js = '''
        function switchDiscoveryTab(panelId, btnElement) {
            ['searchPanel', 'latestActPanel', 'latestTaskPanel'].forEach(id => {
                document.getElementById(id).classList.add('hidden');
            });
            document.getElementById(panelId).classList.remove('hidden');
            
            document.querySelectorAll('.discovery-tab').forEach(el => {
                el.classList.replace('text-indigo-600', 'text-slate-500');
                el.classList.replace('border-indigo-600', 'border-transparent');
                el.classList.remove('bg-white');
            });
            
            btnElement.classList.replace('text-slate-500', 'text-indigo-600');
            btnElement.classList.replace('border-transparent', 'border-indigo-600');
            btnElement.classList.add('bg-white');
        }

        // Hook to switch tab and hide right panel when specific operations run
        const originalSearchActivity = searchActivity;
        searchActivity = async function() {
            await originalSearchActivity();
            document.getElementById('warResultInfo').innerHTML = document.getElementById('warResultInfo').innerText;
        };
        const originalLoadLatestTasks = loadLatestTasks;
        loadLatestTasks = async function() {
            await originalLoadLatestTasks();
            document.getElementById('latestTaskInfo').innerHTML = document.getElementById('latestTaskInfo').innerText;
        };
'''
script_block = script_block.replace('</script>', custom_js + '\n    </script>')

# Remove closing tags from demo to properly append script
demo = demo.split('</body>')[0]

with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(demo + script_block + '\\n</body>\\n</html>')
print('Successfully overhauled index.html')