# Collaborating with Git

This guide provides instructions for collaborators on how to use Git for version control in our project.

## Setting Up Git

Before you start, make sure you have Git installed on your machine and have access to the repository. If you're new to Git, you might want to [set up Git](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup) with your username and email.

## Cloning the Repository

If you haven't already, clone the repository to your local machine:

```
git clone https://github.com/Hy77/Selfie_drawing_ur3.git
```

## Workflow

1. **Check Status:**
   Before making changes, it's good practice to check the status of your repository:

   ```
   git status
   ```

2. **Pull Changes:**
   Always pull the latest changes from the remote repository to ensure your local copy is up to date:

   ```
   git pull origin main
   ```

3. **Make Changes:**
   Make your changes to the code or files in the repository.

4. **Stage Changes:**
   Once you're done with your changes, stage them for commit:

   ```
   git add .
   ```

5. **Commit Changes:**
   Commit your staged changes with a descriptive message:

   ```
   git commit -m "Your commit message"
   ```

6. **Push Changes:**
   Push your commits to the remote repository:

   ```
   git push origin main
   ```

## Resolving Conflicts

If you encounter conflicts when pulling or pushing changes, you'll need to resolve them manually. Git will indicate which files have conflicts. Open those files and make the necessary adjustments, then stage, commit, and push the changes.

## Best Practices

- Commit often with meaningful messages.
- Pull changes from the remote repository before starting your work and before pushing your changes.
- Coordinate with your team to avoid conflicting changes.
