<?php
require_once('../../../main/auth.php');
include ('../../../connect.php'); ?>
<!DOCTYPE html>
<html lang="en">
<link rel="stylesheet" href="../css/bootstrap.min.css">
  <script src="../../../js/jquery.min.js"></script>
  <script src="../../../js/bootstrap.min.js"></script>
<head>
<meta charset="utf-8">
<title>edit lab</title>
<script src="../ckeditor.js"></script>
<script src="js/sample.js"></script>

</head>
<header class="header clearfix" style="background-color: #3786d6;">
    <?php include('nav.php'); ?>   
  </header>
<body id="main">

<h3><?php echo $_GET["test"]; ?></h3>

<form action="../../savex.php" method="POST">
<input type="hidden" name="id" value="<?php echo $_GET["id"]; ?>">

<textarea id="editor" name="mytable">

</textarea>
 <button class="btn btn-success">update lab with the new form</button>
        </form>
<script>
initSample();
</script>

</body>
</html>
