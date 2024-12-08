<?php 
include('../connect.php');
$a=$_POST['name'];
$result = $db->prepare("SELECT * FROM professions WHERE `name`=:a");
        $result->bindParam(':a', $a);
        $result->execute();
        $rowcountt = $result->rowcount();
        if ($rowcountt>0) {
	header("location: professional.php?response=8");
}
  if ($rowcountt==0) {    

$sql = "INSERT INTO professions (name) VALUES ('$a')";
$q = $db->prepare($sql);
$q->execute();


header("location: professional.php?response=9&name=$a");
?>
<?php } ?>