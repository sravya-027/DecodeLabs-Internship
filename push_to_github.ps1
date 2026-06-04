# PowerShell script to push the project to GitHub
# Usage: run this after installing Git and configuring your GitHub credentials

Set-Location -Path "c:\Users\SRAVYA\OneDrive\文档\pandasApp"

# Optional: set your identity if not already configured
# git config --global user.name "Your Name"
# git config --global user.email "you@example.com"

git init
git add .
git commit -m "Initial commit: Data Science internship project"
git branch -M main
git remote add origin https://github.com/sravya-027/DecodeLabs-Internship.git
git push -u origin main

Write-Host "If push failed, ensure Git is installed and you have permission to push to the repo." -ForegroundColor Yellow
