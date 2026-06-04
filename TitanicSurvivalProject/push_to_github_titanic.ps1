# PowerShell script to initialize and push the Titanic project to GitHub.
# Usage: run this after installing Git and creating a GitHub repository.

Set-Location -Path "c:\Users\SRAVYA\OneDrive\文档\pandasApp\TitanicSurvivalProject"

# Uncomment and set these if not already configured:
# git config --global user.name "Your Name"
# git config --global user.email "you@example.com"

git init
git add .
git commit -m "Initial commit: Titanic Survival Prediction internship project"
git branch -M main
# Replace the URL below with your GitHub repository URL:
git remote add origin https://github.com/your-username/TitanicSurvivalProject.git
git push -u origin main

Write-Host "If push failed, ensure Git is installed and the remote repo URL is correct." -ForegroundColor Yellow
