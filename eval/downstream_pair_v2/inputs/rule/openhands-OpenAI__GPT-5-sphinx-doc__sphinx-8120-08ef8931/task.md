<uploaded_files>
/workspace/sphinx-doc__sphinx__3.3
</uploaded_files>

I've uploaded a python code repository in the directory sphinx-doc__sphinx__3.3. Consider the following issue description:

<issue_description>
locale/<language>/LC_MESSAGES/sphinx.po translation ignored
**Describe the bug**
I read [1] as it should be possible to add a file ``locale/<language>/LC_MESSAGES/sphinx.mo`` to the source dir (same dir as the ``Makefile``) and through that change translations or add additional translation to <language>. 

When I add ``locale/da/LC_MESSAGES/sphinx.po``, with updated entries for ``Fig. %s`` and ``Listing %s``, a ``locale/da/LC_MESSAGES/sphinx.mo`` is created (because of ``gettext_auto_build = True``), but the translations are not used. The translations from the official ``da`` translation [2] is used. Of course ``language = 'da'`` is in ``conf.py``.

[1] http://www.sphinx-doc.org/en/master/usage/configuration.html#confval-locale_dirs
[2] https://github.com/sphinx-doc/sphinx/blob/master/sphinx/locale/da/LC_MESSAGES/sphinx.po

**To Reproduce**
Steps to reproduce the behavior:
```
$ git clone https://github.com/jonascj/sphinx-test-locale-override.git
$ cd sphinx-test-locale-override
$ git checkout 8dea4cd # EDIT: current master showcases workaround, so revert back to see the bug
$ # make python venv however you like
$ pip install sphinx
$ make html
```
Notice that ``locale/da/LC_MESSAGES/sphinx.mo`` has been created. Open ``_build/html/index.html``. 

**Expected behavior**
The caption label for the figure ``figur 1`` should have been ``Foobar 1`` (for the sake of testing) and the caption label for the code block ``Viser 1`` should have been ``Whatever 1`` (again for the sake of testing).

**Your project**
https://github.com/jonascj/sphinx-test-locale-override.git

**Screenshots**
![Screenshot of index.html](https://yapb.in/exUE.png "Screenshot of index.html")

**Environment info**
- OS: Arch Linux 
- Python version: 3.7.3
- Sphinx version: 2.1.2
- Sphinx extensions:  none
- Extra tools: none


</issue_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?
I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Also the development Python environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.
Your task is to make the minimal changes to non-test files in the /workspace/sphinx-doc__sphinx__3.3 directory to ensure the <issue_description> is satisfied.

Follow these phases to resolve the issue:

Phase 1. READING: read the problem and reword it in clearer terms
   1.1 If there are code or config snippets. Express in words any best practices or conventions in them.
   1.2 Hightlight message errors, method names, variables, file names, stack traces, and technical details.
   1.3 Explain the problem in clear terms.
   1.4 Enumerate the steps to reproduce the problem.
   1.5 Hightlight any best practices to take into account when testing and fixing the issue

Phase 2. RUNNING: install and run the tests on the repository
   2.1 Follow the readme
   2.2 Install the environment and anything needed
   2.2 Iterate and figure out how to run the tests

Phase 3. EXPLORATION: find the files that are related to the problem and possible solutions
   3.1 Use `grep` to search for relevant methods, classes, keywords and error messages.
   3.2 Identify all files related to the problem statement.
   3.3 Propose the methods and files to fix the issue and explain why.
   3.4 From the possible file locations, select the most likely location to fix the issue.

Phase 4. TEST CREATION: before implementing any fix, create a script to reproduce and verify the issue.
   4.1 Look at existing test files in the repository to understand the test format/structure.
   4.2 Create a minimal reproduction script that reproduces the located issue.
   4.3 Run the reproduction script to confirm you are reproducing the issue.
   4.4 Adjust the reproduction script as necessary.

Phase 5. FIX ANALYSIS: state clearly the problem and how to fix it
   5.1 State clearly what the problem is.
   5.2 State clearly where the problem is located.
   5.3 State clearly how the test reproduces the issue.
   5.4 State clearly the best practices to take into account in the fix.
   5.5 State clearly how to fix the problem.

Phase 6. FIX IMPLEMENTATION: Edit the source code to implement your chosen solution.
   6.1 Make minimal, focused changes to fix the issue.

Phase 7. VERIFICATION: Test your implementation thoroughly.
   7.1 Run your reproduction script to verify the fix works.
   7.2 Add edge cases to your test script to ensure comprehensive coverage.
   7.3 Run existing tests related to the modified code to ensure you haven't broken anything.

8. FINAL REVIEW: Carefully re-read the problem description and compare your changes with the base commit 795747bdb6b8fb7d717d5bbfc2c3316869e66a73.
   8.1 Ensure you've fully addressed all requirements.
   8.2 Run any tests in the repository related to:
     8.2.1 The issue you are fixing
     8.2.2 The files you modified
     8.2.3 The functions you changed
   8.3 If any tests fail, revise your implementation until all tests pass

Be thorough in your exploration, testing, and reasoning. It's fine if your thinking process is lengthy - quality and completeness are more important than brevity.