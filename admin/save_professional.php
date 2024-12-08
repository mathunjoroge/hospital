<?php 
include('../connect.php');
$a=$_POST['name'];
$b=$_POST['profession'];
$result = $db->prepare("SELECT * FROM professionals WHERE `name`=:a");
        $result->bindParam(':a', $a);
        $result->execute();
        $rowcountt = $result->rowcount();
        if ($rowcountt>0) {
	header("location: professional.php?response=8");
}
  if ($rowcountt==0) {    

$sql = "INSERT INTO professionals (name,profession_id) VALUES ('$a','$b')";
$q = $db->prepare($sql);
$q->execute();


header("location: professional.php?response=9&name=$a");
?>
<?php } ?>